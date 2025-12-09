document.addEventListener('DOMContentLoaded', () => {
    const sourceRadios = document.getElementsByName('source');
    const demoOptions = document.getElementById('demo-options');
    const uploadOptions = document.getElementById('upload-options');
    const customOptions = document.getElementById('custom-options');
    const demoSelect = document.getElementById('demo-select');
    const demoInfo = document.getElementById('demo-info');
    const complexitySlider = document.getElementById('complexity-slider');
    const complexityVal = document.getElementById('complexity-val');
    const loadBtn = document.getElementById('load-btn');
    const extractBtn = document.getElementById('extract-btn');
    const loadStatus = document.getElementById('load-status');
    const loader = document.getElementById('loader');
    const results = document.getElementById('results');
    const equationDisplay = document.getElementById('equation-display');
    const successMsg = document.getElementById('success-msg');

    // Toggle Source
    sourceRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'demo') {
                demoOptions.classList.remove('hidden');
                uploadOptions.classList.add('hidden');
                customOptions.classList.add('hidden');
            } else if (e.target.value === 'upload') {
                demoOptions.classList.add('hidden');
                uploadOptions.classList.remove('hidden');
                customOptions.classList.add('hidden');
            } else {
                demoOptions.classList.add('hidden');
                uploadOptions.classList.add('hidden');
                customOptions.classList.remove('hidden');
            }
        });
    });

    // Update Demo Info
    demoSelect.addEventListener('change', (e) => {
        if (e.target.value === 'physics') {
            demoInfo.textContent = "Hidden Law: y = 2.5 * x^2 + cos(x)";
        } else {
            demoInfo.textContent = "Hidden Rule: Income > 6 AND Debt < 3";
        }
    });

    // Slider Value
    complexitySlider.addEventListener('input', (e) => {
        complexityVal.textContent = e.target.value;
    });

    // Load Model
    loadBtn.addEventListener('click', async () => {
        loadStatus.textContent = "Loading...";
        loadStatus.className = "status";
        
        let source = document.querySelector('input[name="source"]:checked').value;
        const formData = new FormData();
        
        if (source === 'demo') {
            formData.append('source', 'demo');
            formData.append('demo_type', demoSelect.value);
        } else if (source === 'upload') {
            formData.append('source', 'upload');
            const fileInput = document.getElementById('model-upload');
            if (fileInput.files.length === 0) {
                loadStatus.textContent = "Please select a file first.";
                return;
            }
            formData.append('file', fileInput.files[0]);
            formData.append('arch_type', document.getElementById('arch-select').value);
        } else if (source === 'custom') {
            formData.append('source', 'custom_code');
            const fileInput = document.getElementById('custom-upload');
            const code = document.getElementById('custom-code').value;
            const className = document.getElementById('custom-class-name').value;
            
            if (fileInput.files.length === 0 || !code || !className) {
                loadStatus.textContent = "Please fill all fields (Code, Class Name, File).";
                return;
            }
            formData.append('file', fileInput.files[0]);
            formData.append('model_code', code);
            formData.append('class_name', className);
        }

        try {
            const response = await fetch('/api/load_model', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                loadStatus.textContent = "✅ Model Loaded Successfully";
                loadStatus.classList.add('success');
                extractBtn.disabled = false;
            } else {
                loadStatus.textContent = "❌ Error: " + data.error;
            }
        } catch (err) {
            loadStatus.textContent = "❌ Connection Error";
            console.error(err);
        }
    });

    // Extract
    extractBtn.addEventListener('click', async () => {
        loader.classList.remove('hidden');
        results.classList.add('hidden');
        extractBtn.disabled = true;
        
        const mode = document.getElementById('mode-select').value;
        const complexity = complexitySlider.value;

        try {
            const response = await fetch('/api/extract', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    mode: mode,
                    complexity: parseFloat(complexity)
                })
            });
            const data = await response.json();

            loader.classList.add('hidden');
            extractBtn.disabled = false;
            results.classList.remove('hidden');

            if (data.success) {
                equationDisplay.textContent = data.result;
                if (data.note) {
                    successMsg.textContent = data.note;
                    successMsg.classList.remove('hidden');
                } else {
                    successMsg.classList.add('hidden');
                }
            } else {
                equationDisplay.textContent = "Error: " + data.error;
            }
        } catch (err) {
            loader.classList.add('hidden');
            extractBtn.disabled = false;
            console.error(err);
        }
    });
});
