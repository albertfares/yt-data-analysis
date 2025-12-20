document.addEventListener('DOMContentLoaded', function () {
    const chartId = 'sunburstChart';
    const container = document.getElementById(chartId);

    if (!container) return;

    fetch('sunburst_data.json')
        .then(response => response.json())
        .then(data => {
            const plotData = data.data;
            const plotLayout = data.layout;

            // Remove hardcoded size to allow responsiveness
            delete plotLayout.width;
            delete plotLayout.height;
            plotLayout.autosize = true;

            // Adapt to dark theme
            plotLayout.paper_bgcolor = 'rgba(0,0,0,0)';
            plotLayout.plot_bgcolor = 'rgba(0,0,0,0)';
            if (plotLayout.font) {
                plotLayout.font.color = 'rgba(255,255,255,0.9)';
            } else {
                plotLayout.font = { color: 'rgba(255,255,255,0.9)' };
            }

            // Function to calculate contrast color (Black or White) based on background hex
            function getContrastColor(hex) {
                // If no color or invalid, default to white for dark theme
                if (!hex || !hex.startsWith('#')) return 'rgba(255,255,255,0.9)';

                const r = parseInt(hex.substr(1, 2), 16);
                const g = parseInt(hex.substr(3, 2), 16);
                const b = parseInt(hex.substr(5, 2), 16);

                // Calculate luminance
                const yiq = ((r * 299) + (g * 587) + (b * 114)) / 1000;

                // Threshold of 160 favors white text on mid-tones (like the brown/orange)
                return (yiq >= 160) ? 'black' : 'white';
            }

            // Apply specific text colors to sectors if colors are available
            if (plotData.length > 0 && plotData[0].marker && plotData[0].marker.colors) {
                const colors = plotData[0].marker.colors;
                const textColors = colors.map(c => getContrastColor(c));

                if (!plotData[0].insidetextfont) {
                    plotData[0].insidetextfont = {};
                }
                plotData[0].insidetextfont.color = textColors;

                // Also ensure outside text is visible (defaulting to global font usually works, but just in case)
                // plotData[0].outsidetextfont = { color: 'rgba(255,255,255,0.9)' };
            }

            // Ensure transparent background to match glassmorphism if desired, 
            // but the sunburst colors might need a specific background.
            // The original has 'paper_bgcolor': 'white', 'plot_bgcolor': '#E5ECF6'
            // We might want to make it transparent to blend with the site theme?
            // The site uses "glass" class which is likely dark/translucent.
            // But if the sunburst needs white text/black text contrast, changing bg might break readability.
            // Given "sunburst_custom_palette" title, it probably has specific colors.
            // Let's keep original colors for now to ensure correctness, user can ask to style it later.

            // However, the original layout has: "paper_bgcolor": "white"
            // If the website is dark mode (glass usually implies dark/blur), a big white square will look bad.
            // Let's check style.css to see the theme.

            const config = {
                responsive: true,
                displayModeBar: false // Cleaner look
            };

            Plotly.newPlot(chartId, plotData, plotLayout, config);
        })
        .catch(error => console.error('Error loading sunburst data:', error));
});
