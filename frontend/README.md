# FoodieLens - React Frontend

A modern, responsive React web interface for food classification and analysis, designed to match the FoodieLens concept with a beautiful green-themed UI.

## Features

✅ **Home Page** - Drag & drop image upload with instant analysis
✅ **Analysis Results** - Detailed predictions with confidence scores and nutrition info
✅ **Scan History** - Browse and manage previous food analysis scans
✅ **User Profile** - Statistics, achievements, and preferences
✅ **Responsive Design** - Works seamlessly on desktop, tablet, and mobile
✅ **Real-time Predictions** - Connects to FastAPI backend for AI predictions

## Tech Stack

- **React 18** - UI library
- **Vite** - Fast build tool and dev server
- **CSS3** - Modern styling with variables and flexbox/grid
- **Fetch API** - Backend communication

## Installation

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file (optional):
```
VITE_API_URL=http://localhost:8000
```

## Running the Development Server

```bash
npm run dev
```

The app will start at `http://localhost:3000`

The Vite config automatically proxies API calls from `/api/*` to `http://localhost:8000/*`

## Building for Production

```bash
npm run build
```

The optimized build will be in the `dist/` directory.

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Sidebar.jsx           # Navigation sidebar
│   │   └── AnalysisResults.jsx   # Results display component
│   ├── pages/
│   │   ├── Home.jsx              # Upload and analysis page
│   │   ├── History.jsx           # Scan history page
│   │   └── Profile.jsx           # User profile page
│   ├── styles/
│   │   ├── index.css             # Global styles
│   │   ├── App.css               # App layout
│   │   ├── Sidebar.css           # Sidebar styling
│   │   ├── Home.css              # Home page styling
│   │   ├── AnalysisResults.css   # Results styling
│   │   ├── History.css           # History page styling
│   │   └── Profile.css           # Profile page styling
│   ├── App.jsx                   # Main app component
│   └── main.jsx                  # React entry point
├── index.html                    # HTML template
├── vite.config.js                # Vite configuration
├── package.json                  # Dependencies
└── README.md                     # This file
```

## Color Scheme

- **Primary Green**: #2d9b4e
- **Light Green**: #e8f5e9
- **Text Dark**: #1a1a1a
- **Text Light**: #666666
- **Accent Yellow**: #ffd700

## API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000`

### Required Endpoints

- **POST** `/predict` - Upload image for food classification
  - Request: `FormData` with `file` field
  - Response: `{ success: boolean, predictions: Array, top_prediction: Object }`

## Features in Detail

### 📸 Home Page
- Drag & drop or click to upload food images
- Real-time image preview
- Loading indicator during analysis
- Error handling with user-friendly messages
- "Analyze" and "Clear" buttons

### 🎯 Analysis Results
- Large preview of analyzed image
- Top prediction with confidence badge
- Top 3 predictions with confidence bars
- Typical nutrition information
- Save to Favorites and Share buttons
- Back button to start new scan

### 📜 History
- Grid view of all previous scans
- Each card shows:
  - Food image thumbnail
  - Primary prediction
  - Confidence percentage
  - Scan timestamp
  - Related predictions

### 👤 Profile
- User avatar and basic info
- Statistics: Total scans, avg confidence, day streak, favorite food
- Preferences: Notifications, history saving, dark mode
- Achievements: Unlocked badges and coming soon items
- Logout button

## Responsive Design

The app is fully responsive with breakpoints at:
- Desktop: 1200px+
- Tablet: 768px - 1199px
- Mobile: < 768px

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## Future Enhancements

- [ ] User authentication
- [ ] Persistent storage (LocalStorage/Backend DB)
- [ ] Dark mode toggle
- [ ] Advanced nutrition filtering
- [ ] Share to social media
- [ ] Camera capture instead of upload
- [ ] Favorites management
- [ ] Analytics dashboard

## Troubleshooting

**Port 3000 already in use:**
```bash
npm run dev -- --port 3001
```

**Backend not responding:**
- Ensure FastAPI server is running on port 8000
- Check CORS is enabled in backend
- Verify network connectivity

**Build errors:**
```bash
rm -rf node_modules
npm install
npm run build
```

## License

MIT - Feel free to use this project for your food classification app!
