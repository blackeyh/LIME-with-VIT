# 🧠🔍 LIME x Vision Transformer (ViT): Interpreting Emotions from Images

> Understand your model, not just its predictions.  
> This project combines **LIME** and **Vision Transformers** to open the black box of emotion recognition in images.

---

## 🎯 Project Objective

In an age where AI powers decision-making, **interpretability matters**. This project uses **LIME (Local Interpretable Model-agnostic Explanations)** to reveal which parts of an image influenced the predictions of a **Vision Transformer (ViT)** model fine-tuned for **facial emotion detection**.

By combining **powerful transformers** with **human-centric interpretability**, we aim to:
- 📸 Understand **why** the model predicted a specific emotion.
- ✅ Build **trust** and **transparency** in AI.
- 🧪 Explore LIME’s capabilities with modern vision models.

---

## 🧠 Model

- **Model Used**: [`dima806/facial_emotions_image_detection`](https://huggingface.co/dima806/facial_emotions_image_detection) (fine-tuned ViT)
- **Architecture**: Vision Transformer (ViT)
- **Purpose**: Classify facial emotions from static images

---

## 🧰 Tech Stack

- 🧱 PyTorch
- 🤗 Hugging Face Transformers
- 🧠 LIME (for local interpretability)
- 🖼️ PIL & matplotlib (image processing and visualization)

---

## 📂 File Structure

 model/
🔹 (Your fine-tuned ViT model — not included here)
📄 limetesting.py
📄 requirements.txt
📄 README.md
📸 happy.jpg, angry.jpg, tongue.jpg
🖼️ happy_analysis.png, angry_analysis.png, tongue_analysis.png


- `limetesting.py`: Core script to apply LIME on ViT predictions.
- `model/`: Add your custom or pre-trained ViT model here.
- `requirements.txt`: Lists Python dependencies.

---

## 🖼️ Tested Expressions

| Expression     | Input Image        | LIME Visualization         |
|----------------|--------------------|-----------------------------|
| 😄 Happy        | `happy.jpg`        | `happy_analysis.png`       |
| 😠 Angry        | `angry.jpg`        | `angry_analysis.png`       |
| 😝 Tongue Out   | `tongue.jpg`       | `tongue_analysis.png`      |

Example:

Original Image | LIME Explanation  
:-------------------------:|:-------------------------:  
![Tongue](tongue.jpg) | ![Tongue Analysis](tongue_analysis.png)



## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Lime](https://github.com/marcotcr/lime) for interpretability tools.
- Open-source libraries for facial expression analysis.
