import { defineConfig } from "fumadocs-mdx/config";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

export default defineConfig({
  mdxOptions: {
    remarkPlugins: [remarkMath],
    // Primero, antes del syntax highlighter
    rehypePlugins: (v) => [rehypeKatex, ...v],
  },
});
