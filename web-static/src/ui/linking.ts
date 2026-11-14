// Citation chip <-> source card linking: every element with data-label inside ``root`` gets
// .is-linked when its label is the hovered or focused one. Returns the first linked element.

export function linkLabels(root: ParentNode, label: string | null, selector = "[data-label]"): HTMLElement | null {
  let first: HTMLElement | null = null;
  for (const node of root.querySelectorAll<HTMLElement>(selector)) {
    const on = label !== null && node.dataset.label === label;
    node.classList.toggle("is-linked", on);
    if (on && !first) first = node;
  }
  return first;
}
