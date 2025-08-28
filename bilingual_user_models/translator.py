from typing import Iterable, Optional
import os
import re
import tqdm
import threading

from deep_translator import GoogleTranslator


class ConversationTranslator:
    def __init__(self, target_language: str, source_language: str = "auto") -> None:
        self.target_language = target_language
        self.source_language = source_language
        self._translator = GoogleTranslator(
            source=self.source_language, target=self.target_language
            
        )
        self._lock = threading.Lock()

        # Batching parameters (keep requests within safe size limits)
        self._batch_max_items = 50
        self._batch_max_chars = 4000

    def translate_text(self, text: str) -> str:
        if not text:
            return text
        try:
            with self._lock:
                return self._translator.translate(text)
        except Exception as e:
            print(f"Translation failed for text: {repr(text[:50])}... Error: {e}")
            return text

    def translate_texts(self, texts: list[str]) -> list[str]:
        if not texts:
            return texts
        try:
            with self._lock:
                # deep_translator supports batch translation
                return self._translator.translate_batch(texts)  # type: ignore[attr-defined]
        except Exception as e:
            print(f"Batch translation failed (falling back to single): {e}")
            return [self.translate_text(t) for t in texts]

    def _translate_preserving_newline(self, text: str) -> str:
        if text == "":
            return text
        has_newline = text.endswith("\n")
        core = text[:-1] if has_newline else text
        translated = self.translate_text(core) if core.strip() else core
        return translated + ("\n" if has_newline else "")

    def translate_lines(self, lines: Iterable[str]) -> Iterable[str]:
        tag_matcher = re.compile(r"^(HUMAN|ASSISTANT):(.*)$")

        def flush_batch(batch_meta: list[tuple[str, str]], batch_texts: list[str]) -> Iterable[str]:
            if not batch_texts:
                return []
            translations = self.translate_texts(batch_texts)
            for (prefix, newline), translated in zip(batch_meta, translations):
                yield f"{prefix}{translated}{newline}"

        current_meta: list[tuple[str, str]] = []  # (prefix, newline)
        current_texts: list[str] = []
        current_chars = 0

        for line in lines:
            try:
                if line.strip() == "":
                    # Flush any pending batch before yielding blank line
                    yield from flush_batch(current_meta, current_texts)
                    current_meta, current_texts, current_chars = [], [], 0
                    yield line
                    continue

                has_newline = line.endswith("\n")
                core = line[:-1] if has_newline else line
                newline = "\n" if has_newline else ""

                match = tag_matcher.match(core)
                if match:
                    role = match.group(1)
                    content = match.group(2)
                    prefix = f"{role}:"
                    text_to_translate = content
                else:
                    prefix = ""
                    text_to_translate = core

                if text_to_translate.strip() == "":
                    # Non-content line, just rebuild and yield
                    yield f"{prefix}{text_to_translate}{newline}"
                    continue

                # If adding this item would exceed limits, flush first
                if (
                    len(current_texts) >= self._batch_max_items
                    or current_chars + len(text_to_translate) > self._batch_max_chars
                ):
                    yield from flush_batch(current_meta, current_texts)
                    current_meta, current_texts, current_chars = [], [], 0

                current_meta.append((prefix, newline))
                current_texts.append(text_to_translate)
                current_chars += len(text_to_translate)

            except Exception as e:
                print(f"Error processing line: {repr(line[:50])}... Error: {e}")
                yield line

        # Flush any remaining batched items
        yield from flush_batch(current_meta, current_texts)

    def translate_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        out_path = output_path or input_path
        try:
            with open(input_path, "r", encoding="utf-8") as infile:
                lines = infile.readlines()

            translated_lines = list(self.translate_lines(lines))

            with open(out_path, "w", encoding="utf-8") as outfile:
                outfile.writelines(translated_lines)

            return out_path
        except FileNotFoundError:
            print(f"File not found: {input_path}")
            return input_path
        except UnicodeDecodeError:
            print(f"Unable to decode file: {input_path}")
            return input_path
        except PermissionError:
            print(f"Permission denied: {input_path}")
            return input_path
        except Exception as e:
            print(f"Error processing file {input_path}: {e}")
            return input_path

    def _get_out_file(self, file: str, output_path: str | None = None) -> str:
        if output_path:
            return os.path.join(output_path, file)
        name, ext = os.path.splitext(file)
        ext = ext if ext else ".txt"
        return f"{name}_{self.target_language}{ext}"

    def translate_directory(
        self, input_path: str, output_path: str | None = None
    ) -> str:
        import concurrent.futures

        try:
            entries = os.listdir(input_path)
            files = [f for f in entries if os.path.isfile(os.path.join(input_path, f))]
        except OSError as e:
            print(f"Error accessing directory {input_path}: {e}")
            return input_path

        input_abs = os.path.abspath(input_path)
        output_abs = os.path.abspath(output_path) if output_path else None
        same_dir = (output_abs is None) or (output_abs == input_abs)

        if not same_dir and output_path:
            os.makedirs(output_path, exist_ok=True)

        def worker(file_name: str) -> str:
            in_file = os.path.join(input_path, file_name)
            if same_dir:
                name, ext = os.path.splitext(file_name)
                ext = ext if ext else ".txt"
                out_file = os.path.join(input_path, f"{name}_{self.target_language}{ext}")
            else:
                out_file = os.path.join(output_path, file_name)  # type: ignore[arg-type]
            # Use a fresh translator per thread to avoid shared-client issues
            translator = ConversationTranslator(
                target_language=self.target_language,
                source_language=self.source_language,
            )
            return translator.translate_file(in_file, out_file)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(
                tqdm.tqdm(
                    executor.map(worker, files),
                    total=len(files),
                    desc="Translating files",
                )
            )
        return output_path or input_path
