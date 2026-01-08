import markdown
import sys
from pathlib import Path

def main():
    md_path = Path("/Users/timurunaspekov/.gemini/antigravity/brain/ae946ed4-eeb6-4dc8-a93d-81f66cea2c75/walkthrough.md")
    out_path = Path("report.html")
    
    if not md_path.exists():
        print(f"Error: {md_path} not found")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Convert
    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    
    # Add CSS for nice report
    style = """
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
        h1, h2, h3 { color: #2c3e50; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        code { background-color: #f8f8f8; padding: 2px 5px; border-radius: 3px; }
        pre { background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; }
        blockquote { border-left: 4px solid #3498db; margin: 0; padding-left: 15px; color: #555; }
    </style>
    """
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Benchmark Report</title>
        {style}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
        
    print(f"Generated {out_path.absolute()}")

if __name__ == "__main__":
    main()
