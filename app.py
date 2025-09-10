import gradio as gr 
from model import predict

GITHUB_LINK = "https://github.com/samiali12/twitter-sentiment-analysis"
COFFEE_LINK = "https://www.buymeacoffee.com/samiali"

with gr.Blocks(theme=gr.themes.Soft()) as app:
    title = 'Neural Date Translator: An Attention-based Seq2Seq Model'
    gr.Markdown(
        """
            # 🌟 Neural Date Translator: An Attention-based Seq2Seq Model

            A deep learning model using sequence-to-sequence architecture with attention that converts human-written dates (e.g., "21th of August 2016") into standardized machine-readable format (e.g., "2016-08-21").

        """
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                placeholder='Type date e.g (March 13, 2001)',
                lines=1,
                label='Enter date'
            )
            btn = gr.Button("Translate", variant="primary")

        with gr.Column():
            label = gr.Label(label='Translation')

    gr.Markdown(
            f"""
            🔗 **Source Code on [GitHub]({GITHUB_LINK})**  

            ☕ If you like this project, consider [buying me a coffee]({COFFEE_LINK})  

            <a href="{COFFEE_LINK}" target="_blank">
            <img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" 
                alt="Buy Me A Coffee" height="41" width="174">
            </a>
            """
    )

    gr.Markdown("💾 All predictions are stored for analysis.")

    btn.click(predict, inputs=[text], outputs=[label])



if __name__ == '__main__':
    app.launch(share=True)