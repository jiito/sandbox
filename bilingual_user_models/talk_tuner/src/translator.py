from typing import Iterable, Optional
import re

from deep_translator import GoogleTranslator


class ConversationTranslator:
    def __init__(self, target_language: str, source_language: str = "auto") -> None:
        self.target_language = target_language
        self.source_language = source_language
        self._translator = GoogleTranslator(source=self.source_language, target=self.target_language)

    def translate_text(self, text: str) -> str:
        if not text:
            return text
        return self._translator.translate(text)

    def _translate_preserving_newline(self, text: str) -> str:
        if text == "":
            return text
        has_newline = text.endswith("\n")
        core = text[:-1] if has_newline else text
        translated = self.translate_text(core) if core.strip() else core
        return translated + ("\n" if has_newline else "")

    def translate_lines(self, lines: Iterable[str]) -> Iterable[str]:
        tag_matcher = re.compile(r"^(HUMAN|ASSISTANT):(.*)(\n?)$")
        for line in lines:
            match = tag_matcher.match(line)
            if match:
                role = match.group(1)
                content = match.group(2)
                newline = match.group(3)
                if content.strip():
                    translated = self.translate_text(content)
                else:
                    translated = content
                yield f"{role}:{translated}{newline}"
            elif line.strip() == "":
                yield line
            else:
                yield self._translate_preserving_newline(line)

    def translate_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        out_path = output_path or input_path
        with open(input_path, "r", encoding="utf-8") as infile:
            lines = infile.readlines()

        translated_lines = list(self.translate_lines(lines))

        with open(out_path, "w", encoding="utf-8") as outfile:
            outfile.writelines(translated_lines)

        return out_path


