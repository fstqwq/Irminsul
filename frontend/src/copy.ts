let activeRoot: HTMLElement | null = null;
let installed = false;

export function installMathCopy(root: HTMLElement): void {
  activeRoot = root;
  if (installed) return;
  installed = true;
  document.addEventListener("copy", handleCopy);
}

function handleCopy(event: ClipboardEvent): void {
  if (!activeRoot || !event.clipboardData) return;

  const selection = document.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return;

  const range = selection.getRangeAt(0);
  const scopes = selectedElements(activeRoot, range, ".statement, .abstract-body");
  if (!scopes.length) return;
  if (!selectedElements(activeRoot, range, ".math-fragment").length) return;

  const text = normalizeCopiedText(scopes.map((scope) => textFromNode(scope, range)).join("\n"));
  if (!text) return;

  event.preventDefault();
  event.clipboardData.setData("text/plain", text);
}

function selectedElements(root: HTMLElement, range: Range, selector: string): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(selector)).filter((element) => intersects(range, element));
}

function intersects(range: Range, node: Node): boolean {
  try {
    return range.intersectsNode(node);
  } catch {
    return false;
  }
}

function textFromNode(node: Node, range: Range): string {
  if (!intersects(range, node)) return "";

  if (node.nodeType === Node.TEXT_NODE) {
    return textFromTextNode(node as Text, range);
  }

  if (node.nodeType !== Node.ELEMENT_NODE) return "";

  const element = node as HTMLElement;
  if (element.classList.contains("math-fragment")) return textFromMath(element);
  if (element.tagName === "BR") return "\n";

  return Array.from(element.childNodes)
    .map((child) => textFromNode(child, range))
    .join("");
}

function textFromTextNode(node: Text, range: Range): string {
  let start = 0;
  let end = node.data.length;

  if (range.startContainer === node) start = range.startOffset;
  if (range.endContainer === node) end = range.endOffset;

  return node.data.slice(start, end);
}

function textFromMath(element: HTMLElement): string {
  const tex = element.dataset.tex?.trim() || element.textContent?.trim() || "";
  if (!tex) return "";
  if (element.dataset.display === "true") return `\n$$${tex}$$\n`;
  return tex;
}

function normalizeCopiedText(value: string): string {
  return value
    .replace(/\u00a0/g, " ")
    .replace(/[ \t\f\v]+\n/g, "\n")
    .replace(/\n[ \t\f\v]+/g, "\n")
    .replace(/[ \t\f\v]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}
