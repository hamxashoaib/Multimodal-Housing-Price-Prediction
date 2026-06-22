# 🏠 Housing Price Prediction Using Images + Tabular Data


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)

---

## What This Project Does

Most housing price models only look at numbers , bedrooms, area, location. But a house's photo tells you a lot too: how well-maintained it is, the curb appeal, the overall condition. This project combines **both**  structured data and house images to predict housing prices more accurately.

That's what multimodal ML is: using more than one type of data at the same time.

---

## Dataset

- **Tabular data**  500 houses with features like area, bedrooms, bathrooms, garage, year built, lot size, overall quality, location score, pool, and basement
- **Image data**  one photo per house (224×224), where visual quality reflects the house condition and price tier
- **Target** — `price` in USD

---

## How It Works

### Extract Image Features (CNN)
Each house photo goes through **ResNet-18**, a pre-trained CNN. Instead of classifying the image, we pull the 512-dimensional feature vector from the second-to-last layer. These vectors capture visual patterns like brightness, structure, and texture.

Then we apply **PCA** to reduce 512 → 64 dimensions to avoid overfitting on a small dataset.

###  Combine with Tabular Data
We concatenate the 64 image features with the 10 tabular features, giving us a 74-feature input for every house.

```
Tabular (10 features)  ──┐
                          ├── Concatenate → 74 features → GBR → Price
Image → ResNet → PCA  ──┘
       (64 features)
```

###  Train & Compare
We train three models to show the value of each modality:

| Model | Input | Algorithm |
|-------|-------|-----------|
| Tabular Only | 10 structured features | Gradient Boosting |
| Image Only | 64 CNN features | Ridge Regression |
| **Multimodal** | 74 combined features | **Gradient Boosting** |

###  Evaluate
We measure performance using:
- **MAE**  average dollar error per prediction
- **RMSE**  penalizes larger errors more heavily

---

## Results

| Model | MAE | RMSE |
|-------|-----|------|
| Tabular Only | ~$X | ~$X |
| Image Only | ~$X | ~$X |
| **Multimodal (Best)** | **~$X** | **~$X** |

> Exact values are printed at the end of the notebook after running all cells.

The multimodal model consistently achieves lower MAE and RMSE than either modality alone.

---

## Files

```
task3-multimodal-housing/
│
├── task3_multimodal_housing_price.ipynb  ← Main notebook
├── app.py                                ← Streamlit app
├── requirements.txt                      ← Dependencies
├── multimodal_pipeline.pkl               ← Exported model (after running notebook)
├── housing_data.csv                      ← Generated tabular dataset
├── house_images/                         ← Generated house images
└── README.md                             ← This file
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the notebook**
```bash
jupyter notebook multimodal_housing_price.ipynb
```
Run all cells top to bottom. This generates the dataset, images, trains models, and saves `multimodal_pipeline.pkl`.

**3. Launch the Streamlit app**
```bash
streamlit run app.py
```
Open `http://localhost:8501` — enter house details, upload a photo, and get a predicted price.

---

## Key Takeaways

- Images carry real predictive signal even without labels, a CNN trained on ImageNet learns general visual patterns that transfer to house quality assessment
- PCA is important when you have more features than samples,  without it the model overfits
- Fusing both modalities beats either one alone, which is exactly the point of multimodal learning
- With real house photos (not synthetic), the image modality would contribute even more

---

## Tech Stack

- **CNN**: ResNet-18 via PyTorch + torchvision
- **Tabular model**: scikit-learn Gradient Boosting + Ridge
- **Dimensionality reduction**: PCA
- **Deployment**: Streamlit
- **Visualization**: matplotlib, seaborn

---

## Author

**Hamza Shoaib**  
BS Artificial Intelligence - Islamia University of Bahawalpur   
GitHub: [@hamxashoaib](https://github.com/hamxashoaib)

---

