document.addEventListener('DOMContentLoaded', function () {
    const chartId = 'heatmapChart';
    const container = document.getElementById(chartId);

    if (!container) return;

    fetch('heatmap_data.json')
        .then(response => response.json())
        .then(data => {
            const plotData = data.data;
            const plotLayout = data.layout;

            // Remove hardcoded size
            delete plotLayout.width;
            delete plotLayout.height;
            plotLayout.autosize = true;

            // Adapt to dark theme
            plotLayout.paper_bgcolor = 'rgba(0,0,0,0)';
            // Use semi-transparent white for plot background to create "grid lines" through the gaps
            plotLayout.plot_bgcolor = 'rgba(255, 255, 255, 0.4)';

            if (plotLayout.font) {
                plotLayout.font.color = 'rgba(255,255,255,0.9)';
            } else {
                plotLayout.font = { color: 'rgba(255,255,255,0.9)' };
            }

            // Fix axis labels
            const axisCommon = {
                gridcolor: 'rgba(255,255,255,0.1)',
                linecolor: 'rgba(255,255,255,0.3)',
                tickcolor: 'rgba(255,255,255,0.3)',
                zerolinecolor: 'rgba(255,255,255,0.3)',
                color: 'rgba(255,255,255,0.9)'
            };

            if (plotLayout.xaxis) Object.assign(plotLayout.xaxis, axisCommon);
            if (plotLayout.yaxis) Object.assign(plotLayout.yaxis, axisCommon);

            // Hide grid for heatmap if desired, but nice to have faint ones
            // Adjust title color
            if (plotLayout.title) {
                if (typeof plotLayout.title === 'string') {
                    // It's a string in the original HTML layout
                } else if (typeof plotLayout.title === 'object') {
                    plotLayout.title.font = { color: 'white', size: 20 };
                }
            }

            // Fix Colorbar ticks
            // The extracted JSON has colorbar settings deeply nested sometimes

            const config = {
                responsive: true,
                displayModeBar: false
            };

            Plotly.newPlot(chartId, plotData, plotLayout, config);
        })
        .catch(error => console.error('Error loading heatmap data:', error));
});
