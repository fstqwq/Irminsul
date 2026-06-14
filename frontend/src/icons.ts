import { ChevronDown, ChevronUp, createIcons, ExternalLink, Pencil, Search, Settings, X } from "lucide";

const icons = {
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Pencil,
  Search,
  Settings,
  X
};

export type IconName = "chevron-down" | "chevron-up" | "external-link" | "pencil" | "search" | "settings" | "x";

export function icon(name: IconName, className = ""): string {
  const classAttr = className ? ` class="${className}"` : "";
  return `<i data-lucide="${name}"${classAttr} aria-hidden="true"></i>`;
}

export function renderIcons(root: HTMLElement): void {
  createIcons({
    root,
    icons
  });
}
