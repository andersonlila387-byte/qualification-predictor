"""
Sample Data for Testing the Smart Applicant Qualification System
"""

# Sample job requirements
SAMPLE_JOB_REQUIREMENTS = {
    "software_engineer": {
        "position": "Software Engineer",
        "company": "Tech Corp",
        "required_skills": ["python", "javascript", "sql", "git"],
        "required_experience": 3,
        "required_education": 3
    },
    "data_scientist": {
        "position": "Data Scientist",
        "company": "Data Inc",
        "required_skills": ["python", "machine learning", "statistics", "sql"],
        "required_experience": 2,
        "required_education": 4
    },
    "full_stack_developer": {
        "position": "Full Stack Developer",
        "company": "Web Solutions",
        "required_skills": ["javascript", "react", "node.js", "sql", "docker"],
        "required_experience": 4,
        "required_education": 3
    }
}

# Sample adaptability responses (for testing)
SAMPLE_ADAPTABILITY_RESPONSES = {
    "high_score": [
        "I learned Python in 2 months while working on a project. I used online courses and practiced daily.",
        "When our team switched to remote work, I quickly adapted by setting up a home office and learning new collaboration tools.",
        "I approach unfamiliar problems by breaking them down into smaller parts and researching each component systematically.",
        "When I joined a new team, I made an effort to understand each member's working style and communicated openly to ensure smooth collaboration."
    ],
    "medium_score": [
        "I learned some new skills when required for a project.",
        "I have adapted to changes in my work environment when needed.",
        "I try to solve problems by asking questions and researching.",
        "I have worked with different teams over the years."
    ],
    "low_score": [
        "I haven't had to learn many new things quickly.",
        "I don't like changes in my work.",
        "I usually wait for someone to help me with problems.",
        "I prefer working with the same team."
    ]
}

# Sample resume text (for text-based processing)
SAMPLE_RESUME_TEXT = """
JOHN DOE
Software Engineer
john.doe@email.com | (555) 123-4567 | LinkedIn: linkedin.com/in/johndoe

PROFESSIONAL SUMMARY
Experienced Software Engineer with 5 years of experience in developing web applications
and solutions. Proficient in Python, JavaScript, and cloud technologies.

SKILLS
Programming: Python, JavaScript, Java, C++
Web Technologies: React, Node.js, Django, Flask, HTML, CSS
Databases: PostgreSQL, MySQL, MongoDB
Cloud & DevOps: AWS, Docker, Kubernetes, Jenkins, Git
Tools: JIRA, Git, VS Code, Linux

WORK EXPERIENCE
Software Engineer | Tech Corp | Jan 2020 - Present
- Developed REST APIs using Python and Flask
- Implemented microservices using Docker and Kubernetes
- Led a team of 4 developers on an e-commerce platform
- Improved system performance by 40%

Junior Developer | Web Solutions | Jun 2018 - Dec 2019
- Built front-end components using React
- Collaborated with cross-functional teams
- Maintained legacy systems and implemented new features

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2018

CERTIFICATIONS
- AWS Certified Solutions Architect
- Python Programming Certification
"""

# Feature importance from trained model
SAMPLE_FEATURE_IMPORTANCE = {
    "skill_score": 0.85,
    "experience_normalized": 0.72,
    "education_normalized": 0.45,
    "adaptability_score": 0.68
}
