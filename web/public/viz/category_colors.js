// central color palette for categories to ensure consistency across visualizations
(function () {
    const palette = {
        'Entertainment': '#FF3B30',      // Red
        'Comedy': '#FFD60A',             // Yellow
        'People & Blogs': '#FF9F0A',     // Orange
        'Sports': '#30D158',             // Green
        'Music': '#0A84FF',              // Blue
        'Gaming': '#BF5AF2',             // Purple
        'Howto & Style': '#FF375F',      // Pink
        'Film & Animation': '#AC8E68',   // Gold/Brown
        'Science & Technology': '#64D2FF', // Cyan
        'News & Politics': '#8E8E93',    // Grey
        'Education': '#5E5CE6',          // Indigo
        'Pets & Animals': '#D2691E',     // Chocolate
        'Autos & Vehicles': '#00D1D1',   // Teal
        'Travel & Events': '#E0AAFF',    // Light Purple
        'Nonprofits & Activism': '#95A5A6' // Concrete
    };

    // User wants "opaque". These are opaque hexes.

    window.getCategoryColor = function (category) {
        return palette[category] || '#FFFFFF';
    };

    window.categoryColorMap = palette;
    console.log("Category Colors updated (Distinct Palette)");
})();
