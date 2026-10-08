"""
Typer Command-Line Interface -- Phase 4
Cung cap cac lenh terminal de tuong tac voi RAG Learning System.

Cac lenh chinh:
  python -m src.interfaces.cli list-docs
  python -m src.interfaces.cli index <pdf_path> [--force]
  python -m src.interfaces.cli ask "<question>" [--doc-id <id>]
  python -m src.interfaces.cli summarize <doc_id>
  python -m src.interfaces.cli quiz <doc_id> [--topic <t>] [--num <n>] [--interactive]
  python -m src.interfaces.cli flashcards <doc_id> [--topic <t>] [--num <n>]
  python -m src.interfaces.cli delete-doc <doc_id> [--yes]
"""

import sys
import os
import warnings
from typing import Optional
from pathlib import Path

# Dam bao UTF-8 console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Tat warnings khong can thiet
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.core.config import settings
from src.db.store import get_store
from src.pipelines.indexing import index_pdf
from src.pipelines.rag import answer
from src.pipelines.learning import summarize as run_summarize, generate_quiz, generate_flashcards

app = typer.Typer(
    help="🧠 RAG Learning System CLI - Ho tro hoc tap va hoi dap thong minh tu tai lieu.",
    add_completion=False,
)
console = Console()


# ============================================================
# COMMAND: LIST-DOCS
# ============================================================

@app.command("list-docs")
def cmd_list_docs():
    """Liet ke tat ca tai lieu da nap vao he thong Qdrant."""
    with console.status("[cyan]Dang lay danh sach tai lieu...[/cyan]"):
        store = get_store()
        docs = store.list_documents()

    if not docs:
        console.print("[yellow]Chua co tai lieu nao duoc nap trong he thong. Dung lenh 'index' de nap PDF.[/yellow]")
        return

    table = Table(title="📚 Danh sach Tai lieu da nap (Knowledge Base)", show_header=True, header_style="bold magenta")
    table.add_column("STT", style="dim", width=4, justify="center")
    table.add_column("Document ID", style="cyan", width=18)
    table.add_column("Ten File", style="green", min_width=25)
    table.add_column("So Chunks", justify="right", style="yellow")
    table.add_column("Trang cuoi", justify="right", style="dim")

    for idx, d in enumerate(docs, start=1):
        table.add_row(
            str(idx),
            d.get("document_id", ""),
            d.get("filename", "unknown"),
            str(d.get("total_chunks", 0)),
            str(d.get("max_page", 0)),
        )

    console.print(table)


# ============================================================
# COMMAND: INDEX
# ============================================================

@app.command("index")
def cmd_index(
    file_path: str = typer.Argument(..., help="Duong dan den tep tai lieu can nap (PDF, Word .docx, PPTX, TXT, MD)"),
    force: bool = typer.Option(False, "--force", "-f", help="Bat buoc nap lai neu tai lieu da ton tai"),
):
    """Nap tep tai lieu vao he thong (PDF, DOCX, PPTX, TXT, MD -> Chunking -> Embedding -> Qdrant)."""
    p = Path(file_path)
    if not p.exists():
        console.print(f"[bold red]Loi:[/bold red] Khong tim thay tep: [yellow]{file_path}[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Bat dau xu ly tep:[/cyan] [bold]{p.name}[/bold]")
    with console.status("[green]Dang trich xuat, tao embedding va luu vector vao Qdrant...[/green]"):
        try:
            doc_id = index_pdf(p, force_reindex=force)
        except Exception as e:
            console.print(f"[bold red]Nap that bai:[/bold red] {str(e)}")
            raise typer.Exit(code=1)

    console.print(Panel(
        f"[bold green]Nap tai lieu thanh cong![/bold green]\n"
        f"File: [bold]{p.name}[/bold]\n"
        f"Document ID: [cyan]{doc_id}[/cyan]",
        title="Index Success",
        border_style="green",
    ))


# ============================================================
# COMMAND: ASK
# ============================================================

@app.command("ask")
def cmd_ask(
    question: str = typer.Argument(..., help="Cau hoi cua ban"),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", "-d", help="Gioi han tim trong 1 tai lieu cu the"),
):
    """Dat cau hoi dua tren tai lieu (RAG Q&A voi citations)."""
    filter_info = f"trong tai lieu: [cyan]{doc_id}[/cyan]" if doc_id else "tren toan bo tai lieu"
    console.print(f"[cyan]Dang tim kiem va sinh cau tra loi ({filter_info})...[/cyan]")

    with console.status("[yellow]Dang retrieve, rerank va goi LLM...[/yellow]"):
        try:
            res = answer(query=question, document_id=doc_id)
        except Exception as e:
            console.print(f"[bold red]Loi hoi dap:[/bold red] {str(e)}")
            raise typer.Exit(code=1)

    # In cau tra loi
    console.print()
    console.print(Panel(
        Markdown(res.answer),
        title=f"💡 Cau tra loi cho: \"{question}\"",
        border_style="blue",
        padding=(1, 2),
    ))

    # In bang citations neu co
    if res.citations:
        table = Table(title="📖 Nguon trich dan (Citations)", show_header=True, header_style="bold cyan")
        table.add_column("Marker", style="bold yellow", width=8, justify="center")
        table.add_column("Tep tin", style="green", width=25)
        table.add_column("Trang", width=6, justify="center")
        table.add_column("Trich doan noi dung", style="dim", min_width=40)

        for cit in res.citations:
            snippet = cit.chunk.content.strip().replace("\n", " ")
            if len(snippet) > 100:
                snippet = snippet[:100] + "..."
            table.add_row(
                f"[{cit.marker}]",
                cit.chunk.metadata.filename,
                str(cit.chunk.metadata.page),
                snippet,
            )
        console.print(table)
    else:
        console.print("[dim]Khong co trich dan truc tiep duoc ghep noi.[/dim]")


# ============================================================
# COMMAND: SUMMARIZE
# ============================================================

@app.command("summarize")
def cmd_summarize(
    doc_id: str = typer.Argument(..., help="ID tai lieu can tom tat"),
):
    """In ban tom tat tai lieu (tu dong Single Pass hoac Map-Reduce)."""
    console.print(f"[cyan]Dang tom tat tai lieu [bold]{doc_id}[/bold]...[/cyan]")

    with console.status("[yellow]Dang tong hop noi dung qua LLM...[/yellow]"):
        try:
            res = run_summarize(document_id=doc_id)
        except Exception as e:
            console.print(f"[bold red]Loi tom tat:[/bold red] {str(e)}")
            raise typer.Exit(code=1)

    method_label = "Single Pass (Ngan)" if res.method == "single" else "Map-Reduce (Dai)"
    header = f"📄 Ban tom tat ({method_label} - {res.total_chunks} chunks)"

    console.print()
    console.print(Panel(
        Markdown(res.content),
        title=header,
        border_style="magenta",
        padding=(1, 2),
    ))


# ============================================================
# COMMAND: QUIZ
# ============================================================

@app.command("quiz")
def cmd_quiz(
    doc_id: str = typer.Argument(..., help="ID tai lieu"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Chu de tap trung"),
    num: int = typer.Option(5, "--num", "-n", help="So luong cau hoi"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Che do lam bai thi truc tiep tren terminal"),
):
    """Tao bo cau hoi trac nghiem MCQ tu tai lieu."""
    console.print(f"[cyan]Dang tao {num} cau hoi trac nghiem...[/cyan]")

    with console.status("[yellow]Dang phan tich tai lieu va sinh de thi...[/yellow]"):
        try:
            quiz_set = generate_quiz(document_id=doc_id, topic=topic, n_questions=num)
        except Exception as e:
            console.print(f"[bold red]Loi tao quiz:[/bold red] {str(e)}")
            raise typer.Exit(code=1)

    if not quiz_set.items:
        console.print("[yellow]Khong tao duoc cau hoi nao tu tai lieu nay.[/yellow]")
        return

    console.print(f"\n[bold green]Tao thanh cong {len(quiz_set.items)} cau hoi![/bold green]\n")

    if not interactive:
        # In toan bo de va dap an
        for idx, item in enumerate(quiz_set.items, start=1):
            opts_str = "\n".join([f"    {opt}" for opt in item.options])
            content = f"[bold]{item.question}[/bold]\n\n{opts_str}\n\n[bold green]Dap an dung: {item.answer}[/bold green]\n[dim]Giai thich: {item.explanation}[/dim]"
            console.print(Panel(content, title=f"Cau hoi {idx}", border_style="cyan"))
    else:
        # Che do Interactive lam bai thi truc tiep
        console.print("[bold yellow]=== BAT DAU LAM BAI THI TRAC NGHIEM ===[/bold yellow]\n")
        score = 0
        user_answers = []

        for idx, item in enumerate(quiz_set.items, start=1):
            console.print(f"\n[bold cyan]Cau {idx}: {item.question}[/bold cyan]")
            for opt in item.options:
                console.print(f"  {opt}")

            ans = ""
            while ans not in ("A", "B", "C", "D"):
                ans = typer.prompt("Lua chon cua ban (A/B/C/D)").strip().upper()

            user_answers.append(ans)
            if ans == item.answer:
                score += 1
                console.print("[green]Chinh xac! [/green]")
            else:
                console.print(f"[red]Sai roi! Dap an dung la {item.answer}[/red]")
            console.print(f"[dim]Giai thich: {item.explanation}[/dim]")

        # Tong ket
        total = len(quiz_set.items)
        pct = (score / total) * 100
        console.print("\n" + "=" * 50)
        console.print(Panel(
            f"[bold]KET QUA THI:[/bold]\n"
            f"Diem so: [bold green]{score}/{total}[/bold green] ({pct:.1f}%)\n"
            + ("[bold green]Xuat sac! Ban da nam rat vung kien thuc![/bold green]" if pct >= 80 else "[bold yellow]Can co gang xem lai tai lieu nhe![/bold yellow]"),
            title="Bang Diem",
            border_style="green" if pct >= 50 else "red",
        ))


# ============================================================
# COMMAND: FLASHCARDS
# ============================================================

@app.command("flashcards")
def cmd_flashcards(
    doc_id: str = typer.Argument(..., help="ID tai lieu"),
    topic: Optional[str] = typer.Option(None, "--topic", "-t", help="Chu de tap trung"),
    num: int = typer.Option(8, "--num", "-n", help="So luong the ghi nho"),
):
    """Tao bo the ghi nho (Flashcards) tu tai lieu."""
    console.print(f"[cyan]Dang tao {num} the flashcards...[/cyan]")

    with console.status("[yellow]Dang phan tich va trich xuat khai niem...[/yellow]"):
        try:
            fc_set = generate_flashcards(document_id=doc_id, topic=topic, n_cards=num)
        except Exception as e:
            console.print(f"[bold red]Loi tao flashcards:[/bold red] {str(e)}")
            raise typer.Exit(code=1)

    if not fc_set.cards:
        console.print("[yellow]Khong tao duoc flashcard nao tu tai lieu nay.[/yellow]")
        return

    table = Table(title=f"🗂️ Bo Flashcards ({len(fc_set.cards)} the)", show_header=True, header_style="bold yellow")
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("Mat truoc (Khai niem / Cau hoi)", style="bold cyan", min_width=25)
    table.add_column("Mat sau (Dinh nghia / Giai thich)", style="green", min_width=35)

    for idx, card in enumerate(fc_set.cards, start=1):
        table.add_row(str(idx), card.front, card.back)

    console.print(table)


# ============================================================
# COMMAND: DELETE-DOC
# ============================================================

@app.command("delete-doc")
def cmd_delete_doc(
    doc_id: str = typer.Argument(..., help="ID tai lieu can xoa"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Bo qua buoc xac nhan"),
):
    """Xoa tai lieu va cac vector lien quan khoi he thong Qdrant."""
    if not yes:
        confirm = typer.confirm(f"Ban co chac chan muon xoa tai lieu ID '{doc_id}' khong?")
        if not confirm:
            console.print("[yellow]Huy thao tac xoa.[/yellow]")
            return

    with console.status(f"[red]Dang xoa tai lieu {doc_id}...[/red]"):
        try:
            store = get_store()
            store.delete_document(doc_id)
            console.print(f"[bold green]Da xoa thanh cong tai lieu:[/bold green] [cyan]{doc_id}[/cyan]")
        except Exception as e:
            console.print(f"[bold red]Loi xoa tai lieu:[/bold red] {str(e)}")
            raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
