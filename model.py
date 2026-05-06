from langchain_openai import ChatOpenAI

llm_basic = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
llm_advanced = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))