import express from 'express'
import cors from 'cors'
import { spawn } from 'child_process'
import path from 'path'
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
console.log(__dirname);
const app = express()
const port = 3000


app.use(express.json())
app.use(cors())

app.get('/', (req, res) => {
  res.send('Hello World!')
})


const VENV_DIR = path.join(__dirname, '../notesAgent/.venv'); // Name of your virtual environment directory
console.log(VENV_DIR)
let pythonExecutable;

if (process.platform === 'win32') {
    pythonExecutable = path.join(VENV_DIR, 'Scripts', 'python.exe');
} else {
    pythonExecutable = path.join(VENV_DIR, 'bin', 'python');
}
// Ensure the main.py script is also resolved correctly
const pythonScriptPath = path.join(__dirname,"../notesAgent/main.py");
// console.log(pythonScriptPath)

let messages = [];

app.post('/chat', (req, res)=>{
    const prompt = req.body.prompt;
    messages.push(prompt)
    const messagePackage = JSON.stringify(messages);

    const pythonProcess = spawn(pythonExecutable, [pythonScriptPath, messagePackage])

    let llmOutput = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
        
        llmOutput += data.toString();
        console.log(llmOutput)
        // llmOutput += data.toString();
        // console.log(`Python stdout: ${data.toString().trim()}`); // For debugging
    });
 
    // Listen for data from standard error (stderr) of the Python process
    // pythonProcess.stderr.on('data', (data) => {
    //     errorOutput += data.toString();
    //     console.error(`Python stderr: ${data.toString().trim()}`); // For debugging
    // });

    // Listen for the 'close' event, which indicates the Python process has exited
    pythonProcess.on('close', (code) => {
        if (code === 0) { // Python script exited successfully
            try {
                // Attempt to parse the JSON output from the Python script
                const parsedOutput = JSON.parse(llmOutput);
                messages.push(llmOutput)
                res.json({ success: true, reply: parsedOutput });
            } catch (e) {
                console.error("Error parsing Python JSON output:", e);
                res.status(500).json({ success: false, error: 'Failed to parse JSON output from Python script', reply: llmOutput });
            }
        } else { // Python script exited with an error
            console.error(`Python script exited with code ${code}. Error: ${errorOutput}`);
            res.status(500).json({ success: false, error: 'Python script execution failed', reply: errorOutput });
        }
    });

    // Handle potential errors during spawning (e.g., 'python' command not found)
    pythonProcess.on('error', (err) => {
        console.error('Failed to start Python process:', err);
        res.status(500).json({ success: false, error: 'Failed to start Python process', details: err.message });
    });

})


app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})
