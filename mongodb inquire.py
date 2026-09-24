from pymongo import MongoClient

# 替换为你的实际 MongoDB Atlas 连接 URI
MONGO_URI = "mongodb+srv://user1:1234@cluster0.63cfp.mongodb.net"



client = MongoClient(MONGO_URI)

# 获取所有数据库
databases = client.list_database_names()
print("📂 数据库列表：", databases)
