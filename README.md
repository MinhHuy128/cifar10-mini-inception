# 🌟 CIFAR-10 Image Classification with Custom Mini-InceptionNet

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

## 📌 Project Overview
This repository contains a complete deep learning pipeline for image classification on the CIFAR-10 dataset. Instead of relying on pre-trained models, I designed and trained a **custom Mini-InceptionNet** from scratch. The project demonstrates core computer vision concepts, robust training methodologies, and model deployment via a web interface.

## 🚀 Key Features
* **Custom Architecture**: Built a lightweight version of the Inception network featuring parallel 1x1, 3x3, and 5x5 convolutional branches.
* **Robust Training Pipeline**: Implemented Data Augmentation, `CosineAnnealingLR` scheduler, and SGD with Momentum & Weight Decay.
* **Overfitting Prevention**: Integrated **Model Checkpointing** and **Early Stopping** mechanisms to capture the optimal weights.
* **Experiment Tracking**: Utilized TensorBoard for real-time loss and accuracy visualization.
* **Interactive Web UI**: Deployed the trained model using Gradio for real-time inference.

## 📂 Repository Structure
```text
├── dataset.py         # Data loading and augmentation pipeline
├── model.py           # Mini-InceptionNet architecture definition
├── train.py           # Training loop with Early Stopping & TensorBoard
├── evaluate.py        # Model evaluation and Confusion Matrix generation
├── app.py             # Gradio web interface for real-time inference
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
⚙️ Installation & Usage
1. Clone the repository:

Bash
git clone [https://github.com/your-username/cifar10-mini-inception.git](https://github.com/your-username/cifar10-mini-inception.git)
cd cifar10-mini-inception
2. Install dependencies:

Bash
pip install -r requirements.txt
3. Train the model:

Bash
python train.py
4. Run the Web Interface:

Bash
python app.py
After running, click the local or public URL provided in the terminal to interact with the model.

📊 Results & Evaluation
The model's performance is thoroughly evaluated using metrics such as Precision, Recall, F1-Score, and a Confusion Matrix to analyze class-wise accuracy.

(Add your Confusion Matrix image here later by uploading confusion_matrix.png to the repo and replacing this line with ![Confusion Matrix](confusion_matrix.png))

*Lưu ý: Ở phần `git clone`, bạn nhớ thay `your-username` bằng tên tài khoản GitHub thực tế của bạn nhé.*

---

### Bước 3: Đẩy source code lên GitHub bằng Terminal

Bây giờ thư mục của bạn đã hoàn hảo. Hãy làm theo các lệnh sau trong Terminal (đảm bảo bạn đang đứng ở thư mục dự án):

1. **Khởi tạo Git:**
   `git init`
2. **Thêm tất cả các file vào luồng chờ (Nó sẽ tự động bỏ qua những file trong .gitignore):**
   `git add .`
3. **Lưu lại phiên bản đầu tiên:**
   `git commit -m "Initial commit: Add custom Mini-InceptionNet pipeline, training script with Early Stopping, and Gradio App"`
4. **Tạo Repository trên Web:** Bạn đăng nhập vào [github.com](https://github.com), bấm nút **New** tạo một repository mới tên là `cifar10-mini-inception` (Nhớ để trống, KHÔNG tích chọn tạo thêm README hay .gitignore trên web nữa vì mình đã làm ở máy rồi).
5. **Liên kết máy tính với GitHub và đẩy code lên (Copy 3 dòng GitHub gợi ý dán vào terminal):**
   ```bash
   git branch -M main
   git remote add origin https://github.com/tên-tài-khoản-của-bạn/cifar10-mini-inception.git
   git push -u origin main