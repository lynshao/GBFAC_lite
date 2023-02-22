# GBFAC_lite
The source code of GBAFC with enhaced feature extractor.

Design C of the enhaced feature extractor is adopted, the architecture of which is:

Linear + ReLu + Linear + ReLu + Linear

There is no belief update and successive decoding.

# Well-trained models

This repo provides two well-trained models for noiseless feedback channels and three well-trained models for a noisy feedback channel of 20 dB. The results we obatined are as follows:

Noiseless feedback:
| Feedforwad SNR | Feedback SNR | BLER |
| ------------- | ------------- |  ------------- |
| -1.5 dB  | 100 dB  | ? |
| -1 dB  | 100 dB  | 1.3e-9 |

Noisy feedback:
| Feedforwad SNR | Feedback SNR | BLER |
| ------------- | ------------- |  ------------- |
| -1 dB  | 20 dB  | 2.33e-2 |
| 0 dB  | 20 dB  | 4.05e-5 |
| 1 dB  | 20 dB  | 4e-7 |

To reproduce the results, please run

python main.py --snr1 [input] --snr2 [input] --train 0 --batchSize 100000
