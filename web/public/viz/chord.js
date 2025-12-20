document.addEventListener('DOMContentLoaded', function () {
    fetch('interactive_chord.json')
        .then(response => response.json())
        .then(data => {
            // The JSON from Bokeh usually has a root key (UUID) then the actual item content
            // or it is the item content directly. 
            // Based on the 'head' output seen earlier:
            // { "uuid": { "version": ..., "roots": ... } }
            // Bokeh.embed.embed_item expects the inner object.

            const keys = Object.keys(data);
            if (keys.length === 0) {
                console.error('Chord data is empty');
                return;
            }

            // Assuming the first key is the root ID/UUID
            const item = data[keys[0]];

            // We need to tell Bokeh where to embed it. 
            // If the item itself doesn't have the target ID set to 'chord-diagram', we can pass it.
            // embed_item(item, target_id)

            Bokeh.embed.embed_item(item, "chord-diagram");
        })
        .catch(error => {
            console.error('Error loading chord diagram:', error);
            document.getElementById('chord-diagram').innerText = "Failed to load visualization.";
        });
});
