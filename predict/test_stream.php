<?php
// c:\Users\DELL\Documents\model\predict\test_stream.php

$result_html = "";

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Increase PHP execution time limit
    set_time_limit(300);

    // 1. Prepare cURL
    $ch = curl_init('http://localhost:8000/generate_questions');
    
    // Prepare file upload
    $cFile = new CURLFile($_FILES['resume']['tmp_name'], $_FILES['resume']['type'], $_FILES['resume']['name']);
    $postData = [
        'position' => $_POST['position'],
        'company'  => $_POST['company'],
        'resume'   => $cFile
    ];

    curl_setopt($ch, CURLOPT_POST, 1);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $postData);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true); // Return response as string
    curl_setopt($ch, CURLOPT_TIMEOUT, 300); // 5 minutes timeout
    
    // 2. Execute
    $response = curl_exec($ch);

    if (curl_errno($ch)) {
        $result_html = "<div class='error'>Error: " . curl_error($ch) . "</div>";
    } else {
        // Decode and Display
        $data = json_decode($response, true);
        if (isset($data['questions'])) {
            $result_html .= "<div class='success'><h3>Generated Interview Questions</h3>";
            $result_html .= "<ul class='questions-list'>";
            foreach ($data['questions'] as $q) {
                $result_html .= "<li>" . htmlspecialchars($q) . "</li>";
            }
            $result_html .= "</ul></div>";
        } else {
            $result_html = "<div class='error'>Raw Response: " . htmlspecialchars($response) . "</div>";
        }
    }
    
    curl_close($ch);
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Interview Question Generator</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f7fa; }
        .container { background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 30px; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #34495e; }
        input[type="text"], input[type="file"] { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { background-color: #3498db; color: white; border: none; padding: 12px 20px; border-radius: 4px; cursor: pointer; width: 100%; font-size: 16px; transition: background 0.3s; }
        button:hover { background-color: #2980b9; }
        .result { margin-top: 30px; }
        .questions-list { list-style-type: none; padding: 0; }
        .questions-list li { background: #e8f4f8; margin-bottom: 15px; padding: 15px; border-left: 5px solid #3498db; border-radius: 4px; }
        .error { color: #e74c3c; background: #fadbd8; padding: 15px; border-radius: 4px; }
        .success { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Interview Question Generator</h1>
        
        <form method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="position">Job Position</label>
                <input type="text" id="position" name="position" placeholder="e.g. Senior Software Engineer" value="<?php echo isset($_POST['position']) ? htmlspecialchars($_POST['position']) : ''; ?>" required>
            </div>
            
            <div class="form-group">
                <label for="company">Company Name</label>
                <input type="text" id="company" name="company" placeholder="e.g. Tech Innovations Inc" value="<?php echo isset($_POST['company']) ? htmlspecialchars($_POST['company']) : ''; ?>" required>
            </div>
            
            <div class="form-group">
                <label for="resume">Upload Resume (PDF)</label>
                <input type="file" id="resume" name="resume" accept=".pdf" required>
            </div>
            
            <button type="submit">Generate Questions</button>
        </form>

        <div class="result">
            <?php echo $result_html; ?>
        </div>
    </div>
</body>
</html>