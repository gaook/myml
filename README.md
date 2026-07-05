# 我的深度学习笔记

基于 [Hugo](https://gohugo.io/) + [PaperMod](https://github.com/adityatelange/hugo-PaperMod) 的博客。
本地写 Markdown，push 到 GitHub 后由 Cloudflare Pages 自动构建上线。

## 日常写作三步

```bash
# 1. 新建文章（自动带好 front matter）
hugo new posts/my-new-post.md

# 2. 本地实时预览，浏览器打开 http://localhost:1313
#    -D 表示同时预览草稿(draft: true)
hugo server -D

# 3. 发布上线（提交 + 推送，Cloudflare 自动构建）
./publish.sh "写了一篇新文章"
```

## 首次部署（只做一次）

1. 在 GitHub 创建空仓库并推送本项目。
2. Cloudflare Dashboard → Workers & Pages → 创建 Pages 项目 → 连接该 GitHub 仓库。
3. 构建设置：
   - Framework preset: **Hugo**
   - Build command: `hugo`
   - Build output directory: `public`
   - 环境变量：`HUGO_VERSION = 0.162.1`
4. 部署完成后，把 `hugo.toml` 里的 `baseURL` 改成 Cloudflare 给的正式域名，再 `./publish.sh`。

## 目录说明

- `content/posts/` —— 你的文章（Markdown）
- `hugo.toml` —— 站点配置
- `archetypes/posts.md` —— 新文章模板
- `themes/PaperMod` —— 主题（git submodule，别删）
- `public/` —— 构建产物，已被 gitignore，无需提交
