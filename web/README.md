# Book Shelf Frontend (`web/`)

React + Vite frontend for the book recommendation system.

## Quick Start

```bash
cd web
npm install
npm run dev
```

Default dev URL: `http://localhost:5173`  
Expected backend URL: `http://localhost:6006`

To override backend URL, create `web/.env`:

```bash
VITE_API_URL=http://localhost:6006
```

## Structure

- `src/main.jsx`: React entrypoint
- `src/App.jsx`: app shell, routes, cross-page state, global modals
- `src/api.js`: all frontend API calls
- `src/constants.js`: shared UI constants (placeholder image, search options)
- `src/pages/`: page-level views (`GalleryPage`, `BookshelfPage`, `ProfilePage`)
- `src/components/`: reusable UI components and modals

## Main API Endpoints Used

- `POST /recommend`
- `GET /api/recommend/personal`
- `POST /favorites/add`
- `DELETE /favorites/remove`
- `PUT /favorites/update`
- `GET /favorites/list/{user_id}`
- `GET /user/{user_id}/persona`
- `GET /user/{user_id}/stats`
- `POST /marketing/highlights`
- `POST /chat/completions`
- `POST /books/add`

## Notes

- Backend CORS should allow `http://localhost:5173` in local development.
- This frontend was developed in an AI-assisted workflow: AI helped with parts of UI draft/code, while final architecture decisions, API integration, debugging, and validation were completed by the author.
