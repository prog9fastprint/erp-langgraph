"""
Interactive CLI Runner for FastPrint ERP LangGraph Agent.
Uses the 5-node StateGraph pipeline (router -> extractor -> tool_node -> verifier -> synthesizer).
Provides real-time interactive animated spinner and node execution status feedback.
Run with: python cli.py
"""
import sys
import threading
import time
import uuid
from typing import Any, Dict, Optional

from app.core.config import settings
from app.graph.workflow import run_workflow_sync
from app.services.llm import clean_model_display_name, get_default_model_name


class LiveStepSpinner:
    """
    Terminal spinner providing live animated progress and step-by-step node execution feedback.
    Uses ANSI colors and symbols. Gracefully degrades on non-interactive environments (pipes/mocks).
    """
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    NODE_METADATA = {
        "summarizer": ("Summarizer", "Meringkas konteks percakapan"),
        "router": ("Router", "Mengidentifikasi domain & modul target"),
        "extractor": ("Extractor", "Mengekstrak parameter filter & entitas"),
        "tool_node": ("Tool", "Menghubungi backend ERP Django"),
        "rag_node": ("Knowledge", "Pencarian dokumen SOP & spesifikasi"),
        "verifier": ("Verifier", "Memvalidasi kelayakan data & kuantitas"),
        "synthesizer": ("Synthesizer", "Menyusun jawaban terstruktur"),
    }

    def __init__(
        self,
        initial_msg: str = "Memulai pipeline StateGraph...",
        model_name: Optional[str] = None,
    ):
        self.msg = initial_msg
        self.running = False
        self.is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        self.start_time = time.time()
        self.node_start_time = time.time()
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.model_name = model_name or get_default_model_name()
        self.clean_model = clean_model_display_name(self.model_name)

    def set_model(self, model_name: Optional[str]) -> None:
        if model_name:
            with self.lock:
                self.model_name = model_name
                self.clean_model = clean_model_display_name(model_name)

    def start(self) -> "LiveStepSpinner":
        self.start_time = time.time()
        self.node_start_time = time.time()
        if not self.is_tty:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.thread.start()
        return self

    def _spin_loop(self):
        idx = 0
        while self.running:
            with self.lock:
                elapsed = time.time() - self.node_start_time
                frame = self.FRAMES[idx % len(self.FRAMES)]
                model_tag = f"\033[35m[{self.clean_model}]\033[0m " if self.clean_model else ""
                text = f"\r\033[K\033[36m{frame}\033[0m {model_tag}{self.msg} \033[90m({elapsed:.1f}s)\033[0m"
                sys.stdout.write(text)
                sys.stdout.flush()
            time.sleep(0.08)
            idx += 1

    def step_completed(self, node_name: str, update: Dict[str, Any], acc_state: Dict[str, Any]):
        new_model = update.get("active_model") or acc_state.get("active_model")
        if new_model:
            self.set_model(new_model)
        elapsed = time.time() - self.node_start_time
        meta = self.NODE_METADATA.get(node_name, (node_name.capitalize(), node_name))
        node_title = meta[0]

        detail = ""
        icon = "\033[32m✔\033[0m"

        if node_name == "router":
            d = acc_state.get("detected_domain") or update.get("detected_domain")
            t = acc_state.get("detected_tool") or update.get("detected_tool")
            detail = f": {d} -> {t or 'guardrail'}"
        elif node_name == "extractor":
            p = acc_state.get("extracted_params") or update.get("extracted_params") or {}
            clean_p = {k: v for k, v in p.items() if v is not None and k != "company_id"}
            if clean_p:
                items_str = ", ".join(f"{k}='{v}'" for k, v in clean_p.items())
                detail = f": {items_str}"
            else:
                detail = " (parameter terekstrak)"
        elif node_name == "tool_node":
            tr = acc_state.get("tool_result") or update.get("tool_result")
            err = acc_state.get("error_message") or update.get("error_message")
            if err:
                icon = "\033[31m✖\033[0m"
                detail = f": {str(err)[:40]}..."
            elif tr:
                count = (
                    tr.get("total_count", len(tr.get("entries", [])))
                    if isinstance(tr, dict)
                    else (len(tr) if isinstance(tr, list) else 1)
                )
                detail = f": {count} data ditemukan"
            else:
                detail = " (eksekusi selesai)"
        elif node_name == "verifier":
            status = acc_state.get("verification_status") or update.get("verification_status")
            retries = acc_state.get("retry_count", 0)
            if status == "RETRY":
                icon = "\033[33m↻\033[0m"
                detail = f": Data kosong, mengulang retry {retries}/2..."
            elif status == "EMPTY":
                icon = "\033[33m!\033[0m"
                detail = ": Data kosong (batas retry tercapai)"
            elif status == "ERROR":
                icon = "\033[31m✖\033[0m"
                detail = ": Error verifikasi"
            else:
                detail = f": Status {status or 'VALID'}"
        elif node_name == "synthesizer":
            detail = " (selesai)"

        with self.lock:
            if self.is_tty:
                sys.stdout.write(f"\r\033[K{icon} \033[1m[{node_title}]\033[0m Selesai{detail} \033[90m({elapsed:.1f}s)\033[0m\n")
                sys.stdout.flush()
            self.node_start_time = time.time()
            self.msg = "Melanjutkan langkah berikutnya..."

    def stop(self) -> float:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        total_time = time.time() - self.start_time
        if self.is_tty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
        return total_time


def main():
    print("=" * 68)
    print("🤖 FastPrint ERP — LangGraph StateGraph Assistant CLI")
    print("=" * 68)

    provider = settings.MODEL_PROVIDER
    model_name = settings.OPENROUTER_MODEL if provider == "openrouter" else settings.OPENAI_MODEL
    print(f"• Pipeline     : StateGraph 5-Node (Router -> Extractor -> Tool -> Verifier -> Synthesizer)")
    print(f"• LLM Provider : {provider.upper()} ({model_name})")
    print(f"• ERP Base URL : {settings.ERP_BASE_URL}")
    print(f"• Default Co.  : ID {settings.DEFAULT_COMPANY_ID}")
    print("• Perintah     : 'exit', 'quit', 'q'   -> Keluar dari sesi")
    print("                 'new', 'reset'         -> Mulai thread percakapan baru")
    print("                 'co <id>'              -> Ubah Company ID aktif")
    print("                 'token <jwt>'          -> Set user token JWT (atau 'token clear')")
    print("                 'verbose'              -> Toggle rincian eksekusi pipeline")
    print("-" * 68)

    current_company_id = settings.DEFAULT_COMPANY_ID
    current_token: Optional[str] = settings.ERP_API_TOKEN
    session_id = f"cli_{uuid.uuid4().hex[:8]}"
    verbose = False

    while True:
        try:
            prompt_label = f"\nUser [Co:{current_company_id}] > "
            user_input = input(prompt_label).strip()
            if not user_input:
                continue

            cmd_lower = user_input.lower()
            if cmd_lower in ("exit", "quit", "q"):
                print("\nSampai jumpa! Sesi CLI diakhiri.")
                break

            if cmd_lower in ("new", "reset"):
                session_id = f"cli_{uuid.uuid4().hex[:8]}"
                print(f"\n✨ Sesi percakapan di-reset. Thread ID baru: {session_id}")
                continue

            if cmd_lower == "verbose":
                verbose = not verbose
                print(f"\n⚙️ Mode Verbose: {'AKTIF' if verbose else 'NONAKTIF'}")
                continue

            if cmd_lower.startswith("company ") or cmd_lower.startswith("co "):
                parts = user_input.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    current_company_id = int(parts[1])
                    print(f"\n🏢 Active Company ID diubah ke: {current_company_id}")
                else:
                    print("\n⚠️ Format tidak valid. Gunakan: co <id> (contoh: co 2)")
                continue

            if cmd_lower.startswith("token "):
                token_val = user_input[6:].strip()
                if token_val.lower() == "clear":
                    current_token = None
                    print("\n🔑 User token dikosongkan.")
                else:
                    current_token = token_val
                    print(f"\n🔑 User token diset ({current_token[:15]}...).")
                continue

            print("\nMemproses alur StateGraph...", flush=True)

            spinner = LiveStepSpinner("Menginisialisasi pipeline StateGraph...", model_name=model_name).start()

            def on_finish(node_name: str, update: Dict[str, Any], acc_state: Dict[str, Any]):
                spinner.step_completed(node_name, update, acc_state)

            final_state = run_workflow_sync(
                user_query=user_input,
                user_token=current_token,
                company_id=current_company_id,
                thread_id=session_id,
                on_node_finish=on_finish,
            )

            total_elapsed = spinner.stop()
            if spinner.is_tty:
                print(f"⏱️  Total waktu eksekusi: {total_elapsed:.2f}s\n")

            # Session-sticky company resolution update
            resolved_co = final_state.get("company_id")
            if resolved_co and resolved_co != current_company_id:
                current_company_id = resolved_co

            active_model = final_state.get("active_model") or model_name
            clean_model = clean_model_display_name(active_model)
            domain = final_state.get("detected_domain")
            tool = final_state.get("detected_tool")
            status = final_state.get("verification_status")
            retries = final_state.get("retry_count", 0)
            params = final_state.get("extracted_params") or {}
            answer = (
                final_state.get("final_answer")
                or final_state.get("error_message")
                or "Tidak ada respons yang dihasilkan."
            )

            if verbose:
                print("\n--- [Rincian Eksekusi Pipeline StateGraph] ---")
                print(f"• Model AI      : {active_model} ({clean_model})")
                print(f"• Domain        : {domain}")
                print(f"• Target Tool   : {tool or '(None / Guardrail)'}")
                print(f"• Filter Params : {params}")
                print(f"• Status Verif  : {status} (Retries: {retries})")
                if final_state.get("tool_result"):
                    tr = final_state["tool_result"]
                    records = (
                        tr.get("total_count", len(tr.get("entries", [])))
                        if isinstance(tr, dict)
                        else len(tr)
                        if isinstance(tr, list)
                        else 1
                    )
                    print(f"• Jumlah Data   : {records}")
                print("----------------------------------------------\n")
            else:
                badge = f"[Model: {clean_model}] [Route: {domain} -> {tool or 'guardrail'} | Status: {status or 'DONE'}]"
                print(f"  {badge}")

            print(f"\nAssistant:\n{answer}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSesi dihentikan.")
            break
        except Exception as loop_err:
            print(f"\n❌ Kendala saat menjalankan workflow: {loop_err}")


if __name__ == "__main__":
    main()
