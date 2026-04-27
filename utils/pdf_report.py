import pypandoc
import numpy as np

def generate_pdf(signal, pred, labels, explanation):

    # نبني المحتوى كنص Markdown
    content = "# ECG AI Report\n\n"

    content += "## AI Explanation\n"
    content += f"{explanation}\n\n"

    content += "## Top Predictions\n"
    for i in np.argsort(pred)[::-1][:3]:
        content += f"- {labels[i]} — {pred[i]*100:.2f}%\n"

    content += "\n⚠️ AI-generated report. Please consult a cardiologist.\n"

    # 🔥 نحول النص مباشرة إلى PDF
    output_file = "report.pdf"

    pypandoc.convert_text(
        content,
        to="pdf",
        format="md",
        outputfile=output_file,
        extra_args=['--standalone']
    )

    # 🔥 نرجع bytes
    with open(output_file, "rb") as f:
        pdf_bytes = f.read()

    return pdf_bytes