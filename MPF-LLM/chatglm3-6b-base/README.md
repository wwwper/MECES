
If you decide to switch or replace the model **LLM backbone**, you must adapt its embedding injection/replacement logic accordingly:

* **`chatglm3-6b-base`:**  
  The native `generate()` method in ChatGLM does not natively accept embedding tensors (`inputs_embeds`). To enable embedding replacement, we modified the official `modeling_chatglm.py` file.
  
  > ⚠️ **Important Step for ChatGLM3-6B Users:**  
  > After downloading the official `chatglm3-6b-base` model weights/files, you **must replace** the original `modeling_chatglm.py` in your local model folder with the modified `modeling_chatglm.py` provided in this repository.

* **Other LLM Backbones:**  
  Most standard LLMs support passing `inputs_embeds` natively. You can perform embedding replacement directly without altering the underlying modeling code.
