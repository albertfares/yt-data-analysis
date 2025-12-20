document.addEventListener('DOMContentLoaded', function () {
    const chartId = 'violinChart';
    const container = document.getElementById(chartId);
    const select = document.getElementById('violinMetricSelect');

    if (!container || !select) return;

    fetch('violin_graph_new.json')
        .then(response => response.json())
        .then(data => {
            let plotData = data.data;
            const plotLayout = data.layout;

            // Extract updatemenus buttons
            const updateMenus = plotLayout.updatemenus;
            let buttons = [];

            if (updateMenus && updateMenus.length > 0 && updateMenus[0].buttons) {
                buttons = updateMenus[0].buttons;
            }

            // Remove internal updatemenus from layout
            delete plotLayout.updatemenus;

            // Populate custom select
            buttons.forEach((btn, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = btn.label;
                select.appendChild(option);
            });

            // Default to first option (Diplomatic Strength) or user request?
            // User requested "first dropdown category". 
            // In the data, index 0 is "Diplomatic Strength".
            // We set the initial data to match the first button's args if possible.
            // Plotly's initial 'data' usually matches one of the views.
            // Let's explicitly trigger the update for the first button to be safe.

            // Adapt to dark theme (Glassmorphism)
            delete plotLayout.width;
            delete plotLayout.height;
            plotLayout.autosize = true;
            plotLayout.paper_bgcolor = 'rgba(0,0,0,0)';
            plotLayout.plot_bgcolor = 'rgba(0,0,0,0)';

            if (plotLayout.font) {
                plotLayout.font.color = 'rgba(255,255,255,0.9)';
            } else {
                plotLayout.font = { color: 'rgba(255,255,255,0.9)' };
            }

            if (plotLayout.xaxis) {
                plotLayout.xaxis.gridcolor = 'rgba(255,255,255,0.1)';
                plotLayout.xaxis.linecolor = 'rgba(255,255,255,0.3)';
                plotLayout.xaxis.tickcolor = 'rgba(255,255,255,0.3)';
                plotLayout.xaxis.zerolinecolor = 'rgba(255,255,255,0.3)';
            }
            if (plotLayout.yaxis) {
                plotLayout.yaxis.gridcolor = 'rgba(255,255,255,0.1)';
                plotLayout.yaxis.linecolor = 'rgba(255,255,255,0.3)';
                plotLayout.yaxis.tickcolor = 'rgba(255,255,255,0.3)';
                plotLayout.yaxis.zerolinecolor = 'rgba(255,255,255,0.3)';
            }
            if (plotLayout.legend) {
                plotLayout.legend.font = { color: 'rgba(255,255,255,0.9)' };
                plotLayout.legend.bgcolor = 'rgba(0,0,0,0)';
            }

            const config = {
                responsive: true,
                displayModeBar: false
            };

            // Render initial plot
            Plotly.newPlot(chartId, plotData, plotLayout, config).then(() => {
                // Trigger the first button's update to ensure consistency
                if (buttons.length > 0) {
                    const firstBtn = buttons[0];
                    Plotly.update(chartId, firstBtn.args[0], firstBtn.args[1]);
                    // Set title manually if needed, usually args[1] handles layout updates like title
                }
            });

            // Handle Display Changes
            select.addEventListener('change', function (e) {
                const index = parseInt(e.target.value);
                if (buttons[index]) {
                    const btn = buttons[index];
                    // btn.method is usually "update"
                    // btn.args is [data_update, layout_update]

                    Plotly.update(chartId, btn.args[0], btn.args[1]);
                }
            });
        })
        .catch(error => console.error('Error loading violin data:', error));
});
