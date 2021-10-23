<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Karnezis/PS_Classifier">
    <img src="/public/assets/favicon.png" alt="Logo" height="80">
  </a>

  <h3 align="center">PathoSpotter Classifier</h3>

  <p align="center">
    PathoSpotter is a computational tool built to help pathologists. This repository intends to improve the already operational, yet not optimal, tool entitled Classifier. The objective when concluding this project is to have a fast and secure web page where physicians can upload their digital images of kidney biopsies and obtain a diagnosis from our trained and published neural networks.</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Imagine having a physician colleague, always available to give you his guesses and advices when it comes to diagnosis. This is what this tool is simulating. Our neural networks were trained to see the amyloidosis, hypercellularity, and sclerosis lesions in glomeruli. This tool by no means replaces the need of a trained physician, but intends to be a useful tool in  the nephropathology context.

### Built With

* [JavaScript](https://developer.mozilla.org/pt-BR/docs/Web/JavaScript)
* [Python](https://www.python.org)
* [Tensorflow](https://www.tensorflow.org/?hl=pt-br)

## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

This tutorial assumes you already have a valid Python instalation. If not, please check the websites above for tutorials.

1. Clone the repo
   ```sh
   git clone https://github.com/Karnezis/PS_Classifier.git
   ```
2. Create a virtual environment
   ```sh
   python -m venv env
   ```

3. Activate your environment
   ```sh
   env\\Scripts\\activate.bat
   ```

4. This project is built on top of some libraries. Install them with:
   ```sh
   pip install -r requirements.txt
   ```

5. Now, run our project.
   ```sh
   python .\\src\\main.py

6. Access the website in the link specified.

## Usage

First, upload an image to the system. Then, click the button to submit it. Now wait for the results. _Ta-da!_

## Tips

Empty (for now).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Ellen C. Aguiar - chalegreaguiar@gmail.com  
Ângelo A. Duarte - angeloduarte@uefs.br  
Project Link: [here](https://github.com/Karnezis/PS_Classifier).

<!--## Acknowledgements -->

<!--* []()
* []()
* []()
-->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username

