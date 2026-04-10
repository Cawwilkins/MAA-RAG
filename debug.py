from config import GENERATIVE_MODEL

def debug_llm(llm):
    print("\n================= LLM DEBUG =================")

    # --- MODEL CONFIG ---
    print("\n--- MODEL CONFIG ---")
    try:
        print(llm._model.config)
    except Exception as e:
        print("Could not access model config:", e)

    # --- TOKENIZER ---
    print("\n--- TOKENIZER ---")
    try:
        print("model_max_length:", llm._tokenizer.model_max_length)
        print("truncation_side:", llm._tokenizer.truncation_side)
        print("padding_side:", llm._tokenizer.padding_side)
    except Exception as e:
        print("Could not access tokenizer:", e)

    # --- GENERATION DEFAULTS ---
    print("\n--- GENERATION DEFAULTS ---")
    try:
        print(llm._gen_defaults)
    except Exception as e:
        print("Could not access generation defaults:", e)

    # --- PARAMETER COUNT ---
    print("\n--- MODEL PARAMETERS ---")
    try:
        total = sum(p.numel() for p in llm._model.parameters())
        print(f"Total parameters: {total:,}")
    except Exception as e:
        print("Could not count parameters:", e)

    # --- METADATA (WHAT LLAMAINDEX SEES) ---
    print("\n--- LLAMAINDEX METADATA ---")
    try:
        meta = llm.metadata
        print("context_window:", meta.context_window)
        print("num_output:", meta.num_output)
        print("model_name:", meta.model_name)
    except Exception as e:
        print("Could not access metadata:", e)

    # --- ESTIMATED TOKEN BUDGET ---
    print("\n--- ESTIMATED TOKEN BUDGET ---")
    try:
        ctx = llm.metadata.context_window
        out = llm.metadata.num_output

        print(f"Context window: {ctx}")
        print(f"Reserved for output: {out}")

        usable = ctx - out
        print(f"Approx usable for prompt+retrieval: {usable}")

        if usable < 0:
            print("⚠️ ERROR: Output tokens exceed context window!")
        elif usable < 500:
            print("⚠️ WARNING: Very small usable context")
        else:
            print("✅ Reasonable context budget")
    except Exception as e:
        print("Could not compute token budget:", e)

    print("\n=============================================\n")


if __name__ == "__main__":
    debug_llm(GENERATIVE_MODEL)