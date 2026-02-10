"""
ERGO Demo

This is the main entry point for the demo application.
"""

import gradio as gr

from .models import model_manager
from .ui import (
    CUSTOM_CSS,
    HEADER_HTML,
    INFO_HTML,
    THEME_TOGGLE_JS,
    EXAMPLES,
    run_ergo_example,
    get_example_choices,
    load_example_display,
)


def create_demo():
    """Create and configure the Gradio demo interface."""
    
    with gr.Blocks(title="ERGO Demo") as demo:
        
        # Header with theme toggle overlaid
        with gr.Column(elem_classes=["header-wrapper"]):
            gr.HTML(HEADER_HTML)
            theme_btn = gr.Button(
                "☀️ Day",
                size="sm",
                elem_classes=["theme-toggle-btn"],
            )
            theme_btn.click(fn=None, js=THEME_TOGGLE_JS)
        
        # Main content
        with gr.Row(equal_height=False):
            
            # Left column: Example selection and display
            with gr.Column(scale=2):
                with gr.Group(elem_classes=["section-container"]):
                    gr.HTML('<div class="section-header"><h3>Select Example</h3></div>')
                    
                    # Example selector
                    example_dropdown = gr.Dropdown(
                        choices=get_example_choices(),
                        value=0,
                        label="",
                        container=False,
                        interactive=True,
                    )
                    
                    # Large image display
                    image_display = gr.Image(
                        value=EXAMPLES[0]["image"],
                        label="Input Image",
                        type="filepath",
                        height=400,
                        interactive=False,
                    )
                    
                    # Question display
                    question_display = gr.Markdown(
                        value=f"**Question:**\n\n{EXAMPLES[0]['question']}",
                        elem_classes=["question-display"],
                    )
                    
                    # Run button
                    run_btn = gr.Button(
                        "Run ERGO",
                        variant="primary",
                        size="lg",
                        elem_classes=["run-btn"],
                    )
            
            # Right column: Output
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["section-container"]):
                    gr.HTML('<div class="section-header"><h3>Model Output</h3></div>')
                    
                    output_chatbot = gr.Chatbot(
                        label="",
                        height=None,  # Auto-expand instead of scroll
                        elem_classes=["chatbot-container"],
                        show_label=False,
                    )
                
                with gr.Group(elem_classes=["section-container", "vision-section"]):
                    gr.HTML('<div class="section-header vision-header"><h3>Vision Token Efficiency</h3></div>')
                    vision_stats_html = gr.HTML(value="")
        
        # Info section
        gr.HTML(INFO_HTML)
        
        # Hidden state for example index
        example_state = gr.State(value=0)
        
        # Event handlers
        def on_example_change(idx):
            img_path, question = load_example_display(idx)
            return img_path, f"**Question:**\n\n{question}", idx
        
        example_dropdown.change(
            fn=on_example_change,
            inputs=[example_dropdown],
            outputs=[image_display, question_display, example_state],
        )
        
        run_btn.click(
            fn=run_ergo_example,
            inputs=[example_state],
            outputs=[output_chatbot, vision_stats_html],
        )
    
    return demo


def main():
    """Main entry point."""
    # Load models
    model_manager.load_all()
    
    # Create and launch demo
    demo = create_demo()
    demo.queue()
    demo.launch(debug=True, share=False, css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
