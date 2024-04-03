from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers
from src.helper import *


B_INST, E_INST= "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<<SYS>>\n\n"

instruction = "Convert the following text from English to Hindi: \n\n{text}"

SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT1 + E_SYS
template = B_INST + SYSTEM_PROMPT + instruction + E_INST

prompt = PromptTemplate(input_variables=['text'], template=template)

llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q4_0.bin', 
                    model_type='llama', config={'temperature':0.01, 'max_new_tokens':128})

llmchain = LLMChain(llm=llm, prompt=prompt)

print(llmchain.run("I love Idlis"))