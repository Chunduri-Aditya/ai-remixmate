# Portfolio Integration Guide

## Quick Setup

### Option 1: HTML Embed (any portfolio)
Copy `docs/portfolio-card.html` into your portfolio site, or embed as an iframe:
```html
<iframe src="portfolio-card.html" width="480" height="640" frameborder="0"></iframe>
```

### Option 2: React Component (React/Next.js portfolio)
Copy `docs/portfolio-card.jsx` into your components folder:
```jsx
import RemixMateCard from '@/components/portfolio-card';

export default function Projects() {
  return <RemixMateCard />;
}
```

### Option 3: Link directly from portfolio
Add to your projects section:
- **GitHub**: `https://github.com/Chunduri-Aditya/ai-remixmate`
- **Access Request**: `https://github.com/Chunduri-Aditya/ai-remixmate/issues/new?template=access-request.yml`

## GitHub Setup (Private Repo with Access Requests)

1. Create private repo on GitHub:
   ```bash
   cd ai-remixmate
   git init
   git add .
   git commit -m "Initial commit — AI RemixMate"
   gh repo create Chunduri-Aditya/ai-remixmate --private --source=. --push
   ```

2. Enable Issues on the repo (Settings → Features → Issues ✓)

3. The `.github/ISSUE_TEMPLATE/access-request.yml` will automatically show when someone opens a new issue

4. When a hiring manager requests access:
   - Go to Settings → Collaborators → Add people
   - Add their GitHub username with "Read" access
   - They'll get an email invitation

## How It Works for Hiring Managers

1. They visit your portfolio, see AI RemixMate project card
2. Click "Request Access" → opens GitHub issue form
3. Fill out name, company, role, purpose
4. You get a notification, review the request
5. Grant read access → they can explore the full codebase

## Customization

Update these in both `portfolio-card.html` and `portfolio-card.jsx`:
- GitHub URLs (search for `Chunduri-Aditya`)
- Stats numbers (modules, tests, endpoints)
- Highlight bullet points
- Tags / tech stack
