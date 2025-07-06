---
title: 如何利用 Github 和 Jekyll 快速搭建个人博客
description: 第一次搭建博客，Windows 系统。
date: 2016-07-15 18:30:00 +0800
categories: [Life, Tools]
tags: [guide]
---

*以下操作过程基于 Windows 系统*

### 前期准备

- 注册 [Github](https://github.com/) 帐户
- 下载  [Git](https://git-scm.com/downloads) （关联 Github ，利用命令行操作）
- 下载 [Sublime Text](https://www.sublimetext.com/3) （修改博客模板文档的编辑器）
- 下载 [Cmd Markdown](https://www.zybuluo.com/cmd/) （博文编辑器）

### 快速了解 Github 的主要功能

- Repository：储存代码资源等
- Issue：在组织下的协同工具，开发者可以在此栏目下提问讨论
- Fork：一大核心功能，通过 Fork 探索好玩的开源仓库（找到你一个喜欢的 Jekyll 博客仓库，进行 Fork ）
- Organization：如同线上社群
- Wiki：可以建立开源知识库
- Pull Request：将你认为好的修改发送给源仓库作者

### 配置 Git 本地仓库并关联 Github

1. 查看 Git 版本： `git version`
2. Git 程序初始设定： `git config --global user.name "yourname"`  `git config --global.user.email "your_email@example.com"`  
3. 生成SSH Key：`ssh-keygen -t rsa -C"your_email@example.com"`
4. 获取SSH Key：`cat ~/.ssh/id_rsa.pub`
5. 为 Github 添加 SSH Key ：在网页端的 Settings 下找到 SSH and GPG Key 选项，新增并添加以上地址即可
6. 在电脑中新建一个文件夹作为本地仓库，并在此文件夹下打开 Git Bash，初始化此仓库：`git init`
7. clone 已 fork 的仓库到本地：`git clone <url>`
8. 理清工作目录、暂存区、本地仓库、远程仓库的联系
- 首先，在工作目录下（例如克隆下来的某仓库中）做出修改动作，通过`git add file name`提交到暂存区
- 然后，通过`git commit -m "commit description"`提交暂存文件到本地仓库
- 最后，通过`git push` 将本地仓库同步更新到远程仓库  
9. 以上就是最最简单的大致步骤，还有更多深入的操作可以研究，当然别忘记了`git help`

### Jeykell 博客的个性化配置和基础设定

1. 修改全局设定的文件`_config.yml`：
- title —— 博客标题
- description —— 博客描述
- name —— 博主名字
- email —— 联系邮箱
- bio —— 座右铭 
- favicon —— 图标
- avatar —— 用户博客头像
2. 编辑个人介绍文档`about.html`
3. 为博客绑定域名：在`CNAME`文件中添加自有域名即可，比如：`zhengxixuan.com`
4. 发布第一篇博文： `_posts` 文件夹是博客所有文章储存的位置，注意以下两点：
- 文档命名格式须为 `date-title.md`
- 文档开头须有 YAML 编码信息（此文件夹中一般会有一个 markdown 样式的博文模板）

### 参考文件

- 开智学堂课程《像黑客一样创作》
- [Github Help](https://help.github.com/)
- [Github Guides](https://guides.github.com/)
- [Stackover Flow](https://stackoverflow.com/)

### 学习过程中犯过的傻

1. **懒**：学 Git 桌面版和命令行的时候不认真，那时网页版还够用，直至后来在 Issues 上问了一个蠢问题，浪费大家的时间，才默默地从头开始一步步操作。其实，也不是太难，对吧，只要你踏踏实实去做，不要犯懒。如今，觉得命令行很好用，桌面版没什么感觉，也许 [Source Tree](https://www.sourcetreeapp.com/) 更好用。
2. **思维混乱**：对于我这种从未接触过编程，甚至连电脑都很不懂的小小白，其实会遇到很多大神们完全无法理解的简单问题，比如我一开始把本地仓库随便装在了一个混乱的文件夹，然后半天搞不懂，乱七八糟。所以，要学会让电脑干净整洁，相信这应该是一个 Hacker 的基本素质。
3. **英文不够好**：接触到 Github 之后，觉得英文是很多人学编程的硬伤，因为大部分优质的资源都是英文版。掌握了科学上网和英文阅读技能，世界和视野都会更加开阔，瞬间心情大好，以为自己突然变聪明了。然而，别自欺欺人罢，知道不代表做到，还有很长的路要走。


### ChangeLog

2016-07-15 初稿