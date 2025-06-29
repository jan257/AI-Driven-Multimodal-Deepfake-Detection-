# 🧠 NeuroNova : A Novel AI-Driven Multimodal Deepfake Detection Using Temporal Dynamics, And Gaze Consistency Analysis 

An AI-driven multimodal deepfake detection web application that leverages **Vision Transformers**, **Temporal Neural Networks**, **Spectral Analysis**, and **Eye Movement Detection** to accurately classify media as real or fake. Designed with a futuristic UI, it combines multiple models and deep learning strategies for unparalleled precision.

---

## 🚀 Features

- 🔍 **Multimodal Detection Pipeline**  
  Combines two distinct models — one based on **visual frames** and another on **eye-gaze patterns** — to cross-verify predictions.

- ⚡ **Real-time Inference**  
  Lightweight and optimized inference using pre-trained ResNet and MobileNet models for fast classification.

- 📊 **Interactive Confidence Visualizations**  
  Includes doughnut charts and model confidence breakdown for interpretability.

- 📁 **Media Upload and Seamless UX**  
  Upload video or image files via a sleek interface with GSAP animations and loading screens.

- 🧠 **Softmax Confidence Aggregation**  
  We use softmax scores from both models and aggregate them for **normalized certainty**.

- 🔬 **Frame-based Video Analysis**  
  Videos are split into frames and passed individually through both models to ensure robustness.

- 🌌 **Futuristic UI**  
  Built using **TailwindCSS**, **Chart.js**, and **GSAP** for a modern and engaging interface.

---

## 🧪 Tech Stack

**Frontend**
- HTML5, TailwindCSS
- GSAP for animations
- Chart.js for visualizations

**Backend**
- Python 3.8+, Django
- PyTorch + Torchvision for inference
- OpenCV + PIL for image and video processing

---

## 🧬 Novel Features

✅ **Multimodal Architecture**  
   Fuses two separate deep learning models (frame + gaze) into a single pipeline for better accuracy.

✅ **Softmax-Weighted Confidence Scoring**  
   Instead of raw labels, we calculate softmax probabilities and average them across frames.

✅ **Per-Model Visual Insights**  
   Detailed confidence charts for each model so users can interpret which model was more confident.

✅ **Spectral Gaze Tracking**  
   One model uses subtle eye movement features to differentiate deepfakes from real content.

✅ **Frame Sampling from Videos**  
   Extracts meaningful frames from videos at specified intervals to improve prediction speed and reduce redundancy.

✅ **Modular & Extendable**  
   Easily plug in new models, backends, or processing methods.

---

## 📂 Project Structure

```
deepfake-detector/
├── detector/
│   ├── templates/
│   │   ├── index.html
│   │   ├── upload.html
│   │   ├── result.html
│   │   └── features.html
│   ├── static/
│   ├── deepfake_predictor.py
│   ├── views.py
│   └── urls.py
├── models/
│   ├── frame_model.pth
│   └── gaze_model.pth
├── media/                   # Uploaded files
├── manage.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place pre-trained models
Download and place the following `.pth` files inside the `models/` directory:
- `frame_model.pth`
- `gaze_model.pth`

### 5. Run the development server
```bash
python manage.py runserver
```
Then open [http://localhost:8000](http://localhost:8000)

---

## 📈 Prediction Workflow

1. **User Uploads Media (Image/Video)**
2. **If Video**: Extract frames using OpenCV  
3. **Each Frame**:  
   - Pass through Frame Model (ResNet18)  
   - Pass through Gaze Model (MobileNet)  
   - Collect softmax confidence values  
4. **Aggregate Results**  
   - Majority class wins  
   - Average model confidence for transparency  
5. **Display Prediction + Confidence + Charts**

---

## 📊 Demo

### 📷 Upload Interface
![Upload Page](https://via.placeholder.com/800x400.png?text=Upload+Media+Interface)

### ✅ Result with Confidence
![Prediction Result](https://via.placeholder.com/800x400.png?text=Deepfake+Prediction+Result)

### 📈 Confidence Chart
![Confidence Chart](https://via.placeholder.com/800x400.png?text=Confidence+Doughnut+Chart)

> Replace these URLs with your own screenshots once you deploy!

---

## 🔐 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Django](https://www.djangoproject.com/)
- [TailwindCSS](https://tailwindcss.com/)
- [Chart.js](https://www.chartjs.org/)
- [GSAP](https://greensock.com/gsap/)

---

## 👩‍💻 Author

**Jahnavi P.**  
_Data Analyst • ML Engineer • Full Stack Developer_

> “Detect truth. Fight deception.” — NeuroNova
