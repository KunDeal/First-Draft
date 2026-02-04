import os
import docx
import traceback

def test_read():
    files = [f for f in os.listdir("knowledge_base") if f.endswith(".docx")]
    if not files:
        print("No docx files found")
        return

    filename = files[0]
    filepath = os.path.join("knowledge_base", filename)
    print(f"Testing read of: {filename}")
    
    try:
        doc = docx.Document(filepath)
        print("Successfully opened document")
        print(f"Paragraphs: {len(doc.paragraphs)}")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_read()
