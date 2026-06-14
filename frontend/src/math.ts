import temml from "temml";

type MathToken = {
  type: "math";
  value: string;
  display: boolean;
  open: string;
  close: string;
};

type TextToken = {
  type: "text";
  value: string;
};

type Token = MathToken | TextToken;

export function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => {
    switch (char) {
      case "&":
        return "&amp;";
      case "<":
        return "&lt;";
      case ">":
        return "&gt;";
      case '"':
        return "&quot;";
      default:
        return "&#39;";
    }
  });
}

function isEscaped(text: string, index: number): boolean {
  let slashCount = 0;
  for (let i = index - 1; i >= 0 && text[i] === "\\"; i -= 1) slashCount += 1;
  return slashCount % 2 === 1;
}

function findClosing(text: string, start: number, close: string): number {
  let index = start;
  while (index < text.length) {
    const found = text.indexOf(close, index);
    if (found < 0) return -1;
    if (!isEscaped(text, found)) return found;
    index = found + close.length;
  }
  return -1;
}

function findClosingDollar(text: string, start: number, display: boolean): number {
  const close = display ? "$$" : "$";
  let index = start;
  while (index < text.length) {
    const found = text.indexOf(close, index);
    if (found < 0) return -1;
    if (!isEscaped(text, found) && (display || text[found + 1] !== "$")) return found;
    index = found + close.length;
  }
  return -1;
}

function pushText(tokens: Token[], value: string): void {
  if (!value) return;
  const previous = tokens[tokens.length - 1];
  if (previous?.type === "text") previous.value += value;
  else tokens.push({ type: "text", value });
}

function tokenizeMath(text: string): Token[] {
  const tokens: Token[] = [];
  let cursor = 0;
  let textStart = 0;

  while (cursor < text.length) {
    const slice = text.slice(cursor);

    if (slice.startsWith("\\[")) {
      const close = findClosing(text, cursor + 2, "\\]");
      if (close >= 0) {
        pushText(tokens, text.slice(textStart, cursor));
        tokens.push({
          type: "math",
          value: text.slice(cursor + 2, close),
          display: true,
          open: "\\[",
          close: "\\]"
        });
        cursor = close + 2;
        textStart = cursor;
        continue;
      }
    }

    if (slice.startsWith("\\(")) {
      const close = findClosing(text, cursor + 2, "\\)");
      if (close >= 0) {
        pushText(tokens, text.slice(textStart, cursor));
        tokens.push({
          type: "math",
          value: text.slice(cursor + 2, close),
          display: false,
          open: "\\(",
          close: "\\)"
        });
        cursor = close + 2;
        textStart = cursor;
        continue;
      }
    }

    if (slice.startsWith("$$") && !isEscaped(text, cursor)) {
      const close = findClosingDollar(text, cursor + 2, true);
      if (close >= 0) {
        pushText(tokens, text.slice(textStart, cursor));
        tokens.push({
          type: "math",
          value: text.slice(cursor + 2, close),
          display: true,
          open: "$$",
          close: "$$"
        });
        cursor = close + 2;
        textStart = cursor;
        continue;
      }
    }

    if (text[cursor] === "$" && text[cursor + 1] !== "$" && !isEscaped(text, cursor)) {
      const close = findClosingDollar(text, cursor + 1, false);
      if (close >= 0) {
        pushText(tokens, text.slice(textStart, cursor));
        tokens.push({
          type: "math",
          value: text.slice(cursor + 1, close),
          display: false,
          open: "$",
          close: "$"
        });
        cursor = close + 1;
        textStart = cursor;
        continue;
      }
    }

    cursor += 1;
  }

  pushText(tokens, text.slice(textStart));
  return tokens;
}

function renderText(value: string): string {
  return escapeHtml(value).replace(/\r?\n/g, "<br>");
}

function renderMath(token: MathToken): string {
  const tex = token.value.trim();
  const display = token.display ? "true" : "false";
  const className = token.display ? "math-fragment math-display" : "math-fragment math-inline";
  try {
    const math = temml.renderToString(tex, {
      displayMode: token.display,
      annotate: true,
      throwOnError: false,
      strict: false,
      trust: false
    });
    return `<span class="${className}" data-tex="${escapeHtml(tex)}" data-display="${display}">${math}</span>`;
  } catch {
    return renderText(`${token.open}${token.value}${token.close}`);
  }
}

export function renderMathText(text: string): string {
  return tokenizeMath(text)
    .map((token) => (token.type === "math" ? renderMath(token) : renderText(token.value)))
    .join("");
}
