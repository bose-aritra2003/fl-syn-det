# **Federated Learning for Deepfake Detection**  
🚀 A **Federated Learning** project using **Flower (flwr)** and **TensorFlow** to classify **real vs. fake** images using an **EfficientNetB0** model. The dataset is distributed across **2 clients** and **1 server** for decentralized training.

---

## 📌 **Project Overview**
This project implements **Federated Learning (FL)** for **Deepfake Detection**. The dataset consists of **real** and **fake** images, distributed across multiple clients. The **server** coordinates training without directly accessing client data.

**Frameworks Used:**
- 🌸 **Flower (flwr)** - Federated Learning framework  
- 🧠 **TensorFlow 2.16.1** - Deep learning framework  
- 🖼 **EfficientNetB0** - Pretrained model for feature extraction  

---

## Project Structure

```
.
├── dataset
│   ├── client_1
│   │   ├── test
│   │   │   ├── fake
│   │   │   └── real
│   │   └── train
│   │       ├── fake
│   │       └── real
│   ├── client_2
│   │   ├── test
│   │   │   ├── fake
│   │   │   └── real
│   │   └── train
│   │       ├── fake
│   │       └── real
│   └── server
│       └── test
│           ├── fake
│           └── real
```
---

## ⚙️ **Setup Instructions**

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/bose-aritra2003/fl-syn-det.git
cd fl-syn-det
```

### 2️⃣ **Set Up a Virtual Environment**
```bash
python3 -m venv flower-env
source flower-env/bin/activate
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🎯 **How to Run**
### **Step 1: Start the Server**
```bash
python server.py
```
- The **server** aggregates model updates from clients.

### **Step 2: Start Client 1**
```bash
python client.py --client_id 1
```

### **Step 3: Start Client 2**
```bash
python client.py --client_id 2
```

---

## 🔧 **Configuration**
Modify **`config.py`** to adjust **FL rounds, dataset path, server address**.
```python
# config.py
NUM_ROUNDS = 10  # Number of federated learning rounds
SERVER_ADDRESS = "localhost:8080" # Modify with the actual IPv4 address where the server is running

SERVER_TEST_PATH = "dataset/server/test/"
CLIENT_TRAIN_PATH_TEMPLATE = "dataset/client_{client_id}/train/"
CLIENT_TEST_PATH_TEMPLATE = "dataset/client_{client_id}/test/"
```

---

## 📝 **Dataset Details**
- **Train**: 24,000 **real** + 24,000 **fake** images  
- **Test**: 6,000 **real** + 6,000 **fake** images  
- **Image size**: **64x64 RGB**  
- **Format**: Stored in `train/real`, `train/fake`, `test/real`, `test/fake` directories  

---

## 📌 **Future Improvements**
✅ Expand to **more clients**  
✅ Experiment with **different architectures** (ResNet, Vision Transformers)  
✅ Add **differential privacy** for security  

---

## 🏆 **Contributors**
- **Aritra Bose** - [GitHub](https://github.com/bose-aritra2003)
- **Srijit Kundu** - [GitHub](https://github.com/SrijitK10)
- **Anik Banerjee** - [GitHub](https://github.com/Anik-Banerjee364)
- **Samit Das** - [GitHub](https://github.com/samitdas03)

🚀 **Happy Coding!** 🎯
