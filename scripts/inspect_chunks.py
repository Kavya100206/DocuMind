import sys
from app.database.postgres import SessionLocal
from app.models.document import Document
from app.models.chunk import Chunk

def check_chunks():
    db = SessionLocal()
    chunks = db.query(Chunk).all()
    count = 0
    with open("chunks_output.txt", "w", encoding="utf-8") as f:
        for c in chunks:
            text_lower = c.text.lower()
            if 'title' in text_lower or 'team' in text_lower or 'orbitmind' in text_lower:
                count += 1
                f.write(f"Doc: {c.document_id}, Page: {c.page_number}, Sec: {getattr(c, 'section_name', '')}, Len: {len(c.text)}\n")
                f.write(c.text[:200].replace('\n', ' ') + "\n")
                f.write("-" * 50 + "\n")
                
        f.write(f"Total matching chunks: {count}\n")
    print("Done writing to chunks_output.txt")

if __name__ == "__main__":
    check_chunks()
