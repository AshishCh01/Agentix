// Models often emit LaTeX as \( ... \) and \[ ... \]; remark-math only
// understands $ ... $ and $$ ... $$. Convert, but leave code untouched.
const CODE_SPLIT = /(```[\s\S]*?```|`[^`\n]+`)/g;

export const normalizeMath = (src) => {
  if (!src) return "";
  return src
    .split(CODE_SPLIT)
    .map((part, i) =>
      i % 2 === 1
        ? part
        : part
            .replace(/\\\[([\s\S]*?)\\\]/g, (_, m) => `\n$$\n${m.trim()}\n$$\n`)
            .replace(/\\\(([\s\S]*?)\\\)/g, (_, m) => `$${m.trim()}$`)
    )
    .join("");
};
