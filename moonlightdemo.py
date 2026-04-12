import os
from uuid import uuid4
import torch
import streamlit as st
st.title("Moonlight Assistant")
st.header("Hello! you can interact in natural language to store/retrieve information.")
#st.subheader("Chat demo")
#from google import genai
from qdrant_client import QdrantClient
from FlagEmbedding import BGEM3FlagModel  
from qdrant_client import models  
from qdrant_client.models import VectorParams, Distance, SparseVectorParams
from qdrant_client.models import PointStruct, Filter, MatchValue, FieldCondition, Prefetch, FusionQuery, PointIdsList
from transformers import AutoTokenizer, AutoModelForCausalLM
from haystack.components.routers import TransformersZeroShotTextRouter
labels=["The user is stating information, describing facts, giving notes, or writing something that is not a question.", 
        "The user is asking something, requesting an explanation, or seeking an answer."]
labelsresult={"The user is stating information, describing facts, giving notes, or writing something that is not a question.": "store", 
              "The user is asking something, requesting an explanation, or seeking an answer.": "retrieve"}
metadatatypes=["A fact, idea, definition, decision, or explanation that is generally true and not tied to a specific problem or recent action.", 
        "A problem, error, bug, failure, or obstacle, including diagnosis steps, causes, and resolutions.", 
        "A completed action, finished step, milestone, or recent achievement that shows forward progress."]
metadataresults={"A fact, idea, definition, decision, or explanation that is generally true and not tied to a specific problem or recent action.": "General",
    "A problem, error, bug, failure, or obstacle, including diagnosis steps, causes, and resolutions.": "Troubleshooting",
    "A completed action, finished step, milestone, or recent achievement that shows forward progress.": "Progress"}
metadataask=["A question about a stored fact, idea, definition, or explanation unrelated to problems or achievements.",
        "A question about an error, bug, failure, or problem previously encountered, including fixes or causes.",
        "A question about a completed task, milestone, or achievement previously recorded."]
metadataaskresult={"A question about a stored fact, idea, definition, or explanation unrelated to problems or achievements.": "General",
    "A question about an error, bug, failure, or problem previously encountered, including fixes or causes.": "Troubleshooting",
    "A question about a completed task, milestone, or achievement previously recorded.": "Progress"}
InitialVecs=["Identity: Moonlight is a RAG based memory chatbot. Its purpose is storing and retrieving personal notes, records, and knowledge on demand.",
    "Usage tip: state clearly whether you are saving or looking something up. Use words like error, bug, or fix when logging troubleshooting records, and words like finished or completed for progress entries.",
    "Name origin: the name Moonlight reflects the core philosophy — the assistant only gives back what the user puts in, mirroring input like the moon reflects sunlight.",
    "Current features: retrieves stored text, summarises results to match the user's phrasing, supports multiple knowledge domains through separate collections.",
    "Upcoming features: document uploads and processing, finer control over stored entries such as editing and deleting specific records."]
if "messages" not in st.session_state:
    st.session_state.messages=[]
if "handling" not in st.session_state:
    st.session_state.handling="Auto"
if "outhandling" not in st.session_state:
    st.session_state.outhandling="Summarise"
if "last_stored" not in st.session_state:
    st.session_state.last_stored=None
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
st.session_state.active_collection="Moon"
st.sidebar.title("⚙️ Database Tools")
st.sidebar.markdown("---")
@st.cache_resource
def loading():
    embeddername="BAAI/bge-m3"
    llmmodelname = r"C:\Users\Moon\.cache\huggingface\hub\models--microsoft--Phi-3-mini-4k-instruct\snapshots\0a67737cc96d2554230f90338b163bc6380a2a85"
    routername="MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    database=QdrantClient(url="http://localhost:6333")
    embedder=BGEM3FlagModel(embeddername, use_fp16=True)
    tokenizer=AutoTokenizer.from_pretrained(llmmodelname, cache_dir=r"C:\Users\Moon\.cache\huggingface\hub", trust_remote_code=True)
    llmmodel=AutoModelForCausalLM.from_pretrained(llmmodelname, 
    #cache_dir=r"C:\Users\Moon\.cache\huggingface\hub", 
    trust_remote_code=False, dtype=torch.float16, load_in_4bit=True, 
    #local_files_only=True
    ).to("cuda")
    routerSR=TransformersZeroShotTextRouter(labels=labels, model=routername)
    routerS=TransformersZeroShotTextRouter(labels=metadatatypes, model=routername)
    routerR=TransformersZeroShotTextRouter(labels=metadataask, model=routername)
    return (database, embedder, tokenizer, llmmodel, routerSR, routerS, routerR)
database, embedder, tokenizer, llmmodel, routerSR, routerS, routerR=loading()
routerSR.warm_up()
routerS.warm_up()
routerR.warm_up()
def embed(toembed):
    return embedder.encode(toembed, return_dense=True, return_sparse=True)
summarise_instruction="You are Moonlight Assistant, Use the information in the next line to answer the user's question, Do not add information on your own."
def talkllm(question, instructions, usefulinfo="", length=140):
    talkprompt=tokenizer.apply_chat_template(
            [
            {"role": "system", "content":f"{instructions}\n{usefulinfo}\n"},
            {"role": "user", "content": f" {question}\n"}
            ],
            tokenize=False, #does not tokenize the chat template, keeps it a string (used to view what the chat template looks like while building the system)
            add_generation_prompt=True #adds "assistant" so the LLM Knows when to start answering
            )
    talktokenized=tokenizer(talkprompt, return_tensors="pt") #converts chat template into tokens
    talkinput_ids = talktokenized["input_ids"].to("cuda") #sends the input ids of the tokens to GPU
    talkattention_mask = talktokenized.get("attention_mask").to("cuda") if talktokenized.get("attention_mask") is not None else None #moves attention mask to GPU to ignore padded tokens (if it exists in the futre) Padded tokens can only exist when treating a batch of sequences
    with torch.no_grad(): #disables gradient calculation 
        #talkstopcriteria=StoppingCriteriaList([Shutup(tokenizer)])
        talkoutput=llmmodel.generate(max_new_tokens=length, do_sample=False, input_ids=talkinput_ids, attention_mask=talkattention_mask, #stopping_criteria=talkstopcriteria,
                                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id) #do sample = false always picks the most probable next token (no randomness), padding token set to be the same as eos token given that Phi-3 does not have a special padding token, this allows it to use the end token as the padding token 
        talkgenerated=talkoutput[0][talkinput_ids.size(1):] #slices only the new generated tokens from the LLM
        talkanswer=tokenizer.decode(talkgenerated.tolist(), skip_special_tokens=True) #decodes the answer into readable text while removing special tokens like (<|end|>)
        return(str(talkanswer))
def store(info):
    infoembed=embed(info)
    dense_vector=infoembed['dense_vecs'].tolist()
    sparse_weights=infoembed['lexical_weights']
    token_index=list(sparse_weights.keys())
    token_weight=list(sparse_weights.values())
    sparse_qdrant_vector=models.SparseVector(indices=token_index, values=token_weight)
    vector={"dense": dense_vector, "sparse": sparse_qdrant_vector}
    id=str(uuid4())
    topic=routerS.run(text=info)
    topic=next(iter(topic))
    topic=str(topic)
    topic=metadataresults[topic]
    database.upsert(collection_name=st.session_state.active_collection, wait=True, points=[PointStruct(id=id, vector=vector, payload={"text": info, "tags": [topic]})])
    st.session_state.last_stored=id
    st.success("Stored Successfully.")
def retrieve(query, top_k=5, metadata_filter: str=None):
    queryembedded=embed(query)
    dense_vector=queryembedded['dense_vecs'].tolist()
    sparse_weights=queryembedded["lexical_weights"]
    sparse_qdrant_vector=models.SparseVector(indices=list(sparse_weights.keys()), values=list(sparse_weights.values()))
    #queryvector={"dense": dense_vector, "sparse": sparse_qdrant_vector}
    topic=routerR.run(text=query)
    topic=next(iter(topic))
    topic=str(topic)
    topic=metadataaskresult[topic]
    result=database.query_points(
    collection_name=st.session_state.active_collection,
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=top_k),
        Prefetch(query=sparse_qdrant_vector, using="sparse", limit=top_k)
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=1,
    with_payload=True
).points
#the amount of stored vectors inside the database must be higher than top_k in order for hybrid retrieval (dense and sparse) to work
    info=result[0].payload["text"]
    #below is a temporary accuracy test line
    associatedTag=result[0].payload["tags"]
    if st.session_state.outhandling=="Natural":
        st.write("Here's what i found relating to your question:")
        st.write(info)
        st.session_state.messages.append({"role": "assistant", "content": info})
    #st.write(f"Tagged as: {associatedTag}")
    else:
        llmanswer=talkllm(query, summarise_instruction, info)
        st.write(llmanswer)
        st.session_state.messages.append({"role": "assistant", "content": llmanswer})
def loadcollections():
    response=database.get_collections()
    names=[col.name for col in response.collections]
    return(names)
def addcollection(user):
    names = loadcollections()
    if user not in names:
        database.create_collection(collection_name=user, vectors_config={'dense': VectorParams(size=1024, distance=Distance.COSINE)}, sparse_vectors_config={'sparse': SparseVectorParams()})
        for i in InitialVecs:
            infoembed=embed(i)
            dense_vector=infoembed['dense_vecs'].tolist()
            sparse_weights=infoembed['lexical_weights']
            token_index=list(sparse_weights.keys())
            token_weight=list(sparse_weights.values())
            sparse_qdrant_vector=models.SparseVector(indices=token_index, values=token_weight)
            vector={"dense": dense_vector, "sparse": sparse_qdrant_vector}
            id=str(uuid4())
            topic=routerS.run(text=i)
            topic=next(iter(topic))
            topic=str(topic)
            topic=metadataresults[topic]
            database.upsert(collection_name=st.session_state.active_collection, wait=True, points=[PointStruct(id=id, vector=vector, payload={"text": i, "tags": [topic]})])
        return True
    else:
        return False
def removecollection(user):
    names=loadcollections()
    if user=="Moon":
        database.delete_collection(collection_name="Moon")
        addcollection("Moon")
        return "Moon"
    else:
        database.delete_collection(collection_name=user)
        return "done"
with st.sidebar:
    with st.expander("🗄️ Active Domain"):
        activecollection=st.selectbox(label="Select Domain:", options=loadcollections(), index=(loadcollections()).index(st.session_state.active_collection) if st.session_state.active_collection in loadcollections() else 0, key="collection_selection")
        st.session_state.active_collection=activecollection
    with st.expander("Create Domain"):
        domain_create=st.text_input("New domain name:", key="create_collection")
        if st.button("Create Domain"):
            if domain_create:
                creation=addcollection(domain_create)
                if creation==True:
                    st.success(f"Created {domain_create} successfully")
                else:
                    st.warning(f"Failed to create {domain_create}")
            else:
                st.warning("Please enter a domain name first.")
    with st.expander("Remove Domain"):
        domain_delete=st.selectbox(label="Select Domain", options=loadcollections(), index=0)
        if st.button("Remove Domain"):
            if domain_delete:
                delete=removecollection(domain_delete)
                if delete=="Moon":
                    st.success("Cannot removed Main collection, recreated instead.")
                else:
                    st.success(f"Successfully removed {domain_delete}")
            else:
                st.warning("Please select a domain first.")
    st.markdown("---")
    mode=st.radio("Choose input handling:", ("Store", "Query", "Auto"), index=2)
    if mode:
        st.session_state.handling=mode
    outmode=st.radio("Choose output handling:", ("Summarise", "Natural"), index=1)
    if outmode:
        st.session_state.outhandling=outmode
    undo=st.button("Undo last save")
    if undo and st.session_state.last_stored:
        database.delete(collection_name=st.session_state.active_collection, points_selector=PointIdsList(points=[st.session_state.last_stored]), wait=True)
        st.success("Removed last stored item.")
        st.session_state.last_stored=None
if "active_collection" not in st.session_state:
    st.session_state.active_collection="Moon"
user = st.chat_input("Type 'help' for a list of commands.", key="chat_input")
if user:
    st.session_state.messages.append({"role": "user", "content": user})
    if st.session_state.handling=="Store":
        store(user)
    elif st.session_state.handling=="Query":
        retrieve(user)
    elif st.session_state.handling=="Auto":
        request=routerSR.run(text=user)
        request=next(iter(request))
        request=str(request)
        process=labelsresult[request]
        process=str(process)
        if process.lower()=="store":
            store(user)
        if process.lower()=="retrieve":
            retrieve(user)