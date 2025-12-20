import os
import time

# 1. 设置 HuggingFace 镜像，防止下载模型超时
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 关键修改点：更新了导入路径以匹配最新版 LangChain ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 新的门牌号
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_db():
    print("🚀 开始构建知识库...")

    # --- 第一步：加载多本书 ---
    file_paths = ["data/textbook.txt", "data/workbook.txt"]
    all_documents = []

    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"📖 正在读取: {file_path} ...")
            try:
                loader = TextLoader(file_path, encoding="gb18030")
                docs = loader.load()
                all_documents.extend(docs)
            except Exception as e:
                print(f"❌ 读取错误 {file_path}: {e}")
        else:
            print(f"⚠️ 警告: 找不到文件 {file_path}，跳过。")

    if not all_documents:
        print("❌ 没有读取到任何数据，请检查 data 文件夹！")
        return

    print(f"1. 数据加载完毕，共读取 {len(all_documents)} 个文档对象")

    # --- 第二步：文本切片 (Chunking) ---
    print("2. 正在进行文本切片...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(all_documents)
    print(f"✅ 切片完毕！原书合计被切分为 {len(texts)} 个知识片段")

    # --- 第三步：加载嵌入模型 (Embedding) ---
    print("3. 正在加载嵌入模型 (shibing624/text2vec-base-chinese)...")
    print("   (首次运行会自动下载约 400MB 模型文件，请耐心等待...)")
    # 使用国内镜像源下载模型
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={'device': 'cpu'}  # 建库用CPU足够快，且最稳定
    )

    # --- 第四步：向量化并存储 (Indexing) ---
    print("4. 正在将知识片段转化为向量...")
    db = FAISS.from_documents(texts, embeddings)

    # 保存到本地
    save_path = "data/control_knowledge_base"
    db.save_local(save_path)
    print(f"🎉 成功！知识库已构建完成，保存在: {save_path}")


if __name__ == "__main__":
    create_vector_db()