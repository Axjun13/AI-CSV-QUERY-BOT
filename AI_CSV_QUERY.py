try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    import pandas as pd
    import re
    import math
    import traceback
    print("✅ LangChain and Pandas imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed: pip install langchain langchain-openai")
    exit(1)

pd.set_option("display.width", None)  
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)

def get_llm():
    return ChatOpenAI(
        model="YOUR_MODEL_PATH",
        openai_api_base="YOUR_API_BASE",
        openai_api_key="YOUR_API_KEY",
        temperature=0.9
    )


def load_pipe_file(filepath):
    """Load and clean a pipe-separated file into a DataFrame"""
    df = pd.read_csv(
        filepath,
        sep="|",
        engine="python",
        encoding="utf-8",
        skipinitialspace=True
    )
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def search_rows(df, query, max_results=5):
    """Simple keyword search across all columns"""
    mask = df.apply(
        lambda row: query.lower() in row.astype(str).str.lower().to_string(),
        axis=1
    )
    return df[mask].head(max_results)


def print_in_pages(df, page_size=20):
    """Print DataFrame in chunks with interactive paging"""
    pages = math.ceil(len(df) / page_size)
    for i in range(pages):
        start, end = i * page_size, (i + 1) * page_size
        print(df.iloc[start:end].to_string(index=False))
        if i < pages - 1:
            input(f"\n--- Showing rows {start+1} to {end} of {len(df)} --- Press Enter for next page...\n")


def ask_about_file(llm, df, question, max_retries=3):
    """
    Fully AI-driven assistant:
    - Classifies question as chat or data
    - For data: generates pandas code
    - For pivots: AI infers ALL arguments (index, columns, values, aggfunc)
    """

    def extract_code_block(text: str) -> str:
        code_match = re.search(r"```(?:python)?\n([\s\S]*?)```", text)
        if code_match:
            return code_match.group(1).strip()
        return text.strip()

    # Step 1: Classification
    classification_prompt = f"""
    You are a classifier. User asked: "{question}"
    Respond with only one word:
    - "data" if it's about analyzing DataFrame
    - "chat" if it's general conversation
    Columns: {list(df.columns)}
    """
    decision = llm.invoke([HumanMessage(content=classification_prompt)]).content.strip().lower()

    if "chat" in decision:
        return llm.invoke([HumanMessage(content=question)]).content

    # Step 2: Data question → AI generates pandas code
    error_msg = None
    for attempt in range(max_retries):
        code_prompt = f"""
The user asked: "{question}"

You have a pandas DataFrame called df.
Its columns are: {list(df.columns)}
Its dtypes are:
{df.dtypes.to_string()}

Write ONLY valid pandas code using df to answer the question.
"""


        if error_msg:
            code_prompt += f"\nThe previous code failed with error:\n{error_msg}\nFix it."

        code_raw = llm.invoke([HumanMessage(content=code_prompt)]).content
        code = extract_code_block(code_raw)

        try:
            result = eval(code, {"df": df, "pd": pd})
            if isinstance(result, pd.DataFrame):
                print_in_pages(result)
                return ""
            elif isinstance(result, pd.Series):
                print(result.to_string())
                return ""
            else:
                return str(result)
        except Exception:
            error_msg = traceback.format_exc()

    # Step 3: Fallback → semantic row search
    matches = search_rows(df, question)
    if matches.empty:
        return f"⚠️ Could not resolve your question after {max_retries} attempts."
    context = "\n".join(matches.astype(str).agg(" | ".join, axis=1).tolist())
    fallback_prompt = f"""
    You are a data assistant.
    Answer the question using ONLY this data:
    {context}

    QUESTION: {question}
    """
    return llm.invoke([HumanMessage(content=fallback_prompt)]).content



def test_connection():
    """Test the LLM connection with a simple query"""
    try:
        llm = get_llm()
        message = HumanMessage(content="Hello! Can you tell me what you are?")
        response = llm.invoke([message])
        print("✅ Connection successful!")
        print(f"Response: {response.content}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def chat_with_llm(df=None):
    """Interactive chat with the LLM"""
    llm = get_llm()
    print("🤖 Chat with your local LLM (type 'quit' to exit)")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        try:
            if df is not None:
                response = ask_about_file(llm, df, user_input)
                if response:  # Only print if text response
                    print(f"🤖 LLM (file-based): {response}")
            else:
                message = HumanMessage(content=user_input)
                response = llm.invoke([message])
                print(f"🤖 LLM: {response.content}")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 Testing LLM connection...")
    if test_connection():
        print("\n" + "=" * 50)
        try:
            df = load_pipe_file("input_data_aug.txt")
            print("✅ Pipe-separated file loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load file: {e}")
            df = None
        chat_with_llm(df)
    else:
        print("Please check your LLM server and configuration.")
