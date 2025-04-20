# HGSD-UESTC-SD

# UESTC-SD
A self-collected and annotated data set for modeling complex semantic scenes, copyrighted by UESTC


# Descrition

   <table border="1" style="width: 100%; text-align: center;">
     <caption>Table 1 Detailed comparison of UESTC - SD and regular data sets</caption>
     <tr>
       <th>DataSet</th>
       <th>Occluded - DukeMTMC</th>
       <th>Occluded - Duke - Video</th>
       <th>Mars</th>
       <th>UESTC - SD</th>
     </tr>
     <tr>
       <td>Train(ID/Seq/Images)</td>
       <td>702/15168</td>
       <td>702/292343</td>
       <td>625/8298/509914</td>
       <td>453/19546</td>
     </tr>
     <tr>
       <td>Gallery(ID/Seq/Images)</td>
       <td>1110/17661</td>
       <td>1110/281114</td>
       <td rowspan="2">636/12180/681089</td>
       <td>384/16763</td>
     </tr>
     <tr>
       <td>Query(ID/Seq/Images)</td>
       <td>519/2210</td>
       <td>661/39526</td>
       <td>151/2783</td>
     </tr>
     <tr>
       <td>Camera</td>
       <td>8</td>
       <td>8</td>
       <td>6</td>
       <td>7</td>
     </tr>
     <tr>
       <td>Monitoring Perspective</td>
       <td>Static and fixed</td>
       <td>Static and fixed</td>
       <td>Static and fixed</td>
       <td>Static and fixed</td>
     </tr>
     <tr>
       <td>Occlusion</td>
       <td>Tender</td>
       <td>Tender</td>
       <td>-</td>
       <td>Severe</td>
     </tr>
     <tr>
       <td>Angle</td>
       <td>Smooth inspect</td>
       <td>Smooth inspect</td>
       <td>Smooth inspect</td>
       <td>Multi angle full view</td>
     </tr>
     <tr>
       <td>Size</td>
       <td>-</td>
       <td>Unitary</td>
       <td>-</td>
       <td>Multi scale</td>
     </tr>
     <tr>
       <td>Scene</td>
       <td>Outdoor</td>
       <td>Outdoor</td>
       <td>Outdoor</td>
       <td>Indoor + Outdoor</td>
     </tr>
   </table>


  <table border="1" style="width: 100%; text-align: center;">
    <caption>Table 2 The Tracklets Ratio with a Certain Fraction of Occluded Frames on Query and Gallery Set of Occluded-DukeMTMC-VideoReID and UESTC-SD</caption>
    <tr>
      <th  style="text-align: center" colspan="2">DataSet</th>
      <th>0</th>
      <th>0 - 30</th>
      <th>30 - 50</th>
      <th>50 - 70</th>
      <th>70 - 100</th>
    </tr>
    <tr>
      <td   style="text-align: center" rowspan="2">Occluded - DukeMTMC - VideoReID</td>
      <td>query</td>
      <td>0%</td>
      <td>0%</td>
      <td>3.2%</td>
      <td>12.4%</td>
      <td>80.4%</td>
    </tr>
    <tr>
      <td>gallery</td>
      <td>31.0%</td>
      <td>24.3%</td>
      <td>2.7%</td>
      <td>5.6%</td>
      <td>36.4%</td>
    </tr>
    <tr>
      <td   style="text-align: center" rowspan="2">UESTC - SD</td>
      <td>query</td>
      <td>1.7%</td>
      <td>3.4%</td>
      <td>5.5%</td>
      <td>17.4%</td>
      <td>72.3%</td>
    </tr>
    <tr>
      <td>gallery</td>
      <td>4.3%</td>
      <td>2.6%</td>
      <td>13.1%</td>
      <td>25.3%</td>
      <td>54.7%</td>
    </tr>
  </table>



<table border="1" style="width: 100%; text-align: center;">
  <caption>Table 3 The Ratio of Video Frames with a Certain Fraction of Occlusion on Query and Gallery Set of UESTC-SD and Occluded-DukeMTMC-VideoReID </caption>
  <tr>
    <th>Data Set</th>
    <th>Occluded fraction%</th>
    <th>0-10</th>
    <th>10-20</th>
    <th>20-30</th>
    <th>30-40</th>
    <th>50-60</th>
    <th>60-70</th>
    <th>70-100</th>
  </tr>
  <tr>
    <td rowspan="2" style="white-space: nowrap;"> UESTC-SD </td>
    <td style="white-space: nowrap;">Frames ratio on query</td>
    <td style="white-space: nowrap;">12.8%</td>
    <td style="white-space: nowrap;">11.7%</td>
    <td style="white-space: nowrap;">25.8%</td>
    <td style="white-space: nowrap;">21.6%</td>
    <td style="white-space: nowrap;">13.7%</td>
    <td style="white-space: nowrap;">10.3%</td>
    <td style="white-space: nowrap;">4.1%</td>
  </tr>
  <tr>
    <td style="white-space: nowrap;">Frames ratio on gallery</td>
    <td style="white-space: nowrap;">10.7%</td>
    <td style="white-space: nowrap;">13.5%</td>
    <td style="white-space: nowrap;">23.1%</td>
    <td style="white-space: nowrap;">27.2%</td>
    <td style="white-space: nowrap;">12.3%</td>
    <td style="white-space: nowrap;">9.5%</td>
    <td style="white-space: nowrap;">3.7%</td>
  </tr>
  <tr>
    <td style="white-space: nowrap;">Occluded-DukeMTMC-Video ReID[ST-RFC]</td>
    <td style="white-space: nowrap;">Frames ratio on query</td>
    <td style="white-space: nowrap;">13.5%</td>
    <td style="white-space: nowrap;">11.1%</td>
    <td style="white-space: nowrap;">12.4%</td>
    <td style="white-space: nowrap;">14.8%</td>
    <td style="white-space: nowrap;">15.6%</td>
    <td style="white-space: nowrap;">11.1%</td>
    <td style="white-space: nowrap;">6.6%</td>
  </tr>
  <tr>
    <td style="white-space: nowrap;">Occluded-DukeMTMC-Video ReID[ST-RFC]</td>
    <td style="white-space: nowrap;">Frames ratio on gallery</td>
    <td style="white-space: nowrap;">13.3%</td>
    <td style="white-space: nowrap;">11.2%</td>
    <td style="white-space: nowrap;">12.3%</td>
    <td style="white-space: nowrap;">14.7%</td>
    <td style="white-space: nowrap;">14.6%</td>
    <td style="white-space: nowrap;">11.2%</td>
    <td style="white-space: nowrap;">8.3%</td>
  </tr>
</table>



<!DOCTYPE html>
<html>
<body>
    <table border="1">
        <caption>Table 4 Probability of APO, SPO and semantic occlusion in UESTC-SD</caption>
        <tr>
            <th align="center">Situation</th>
            <th align="center">APO</th>
            <th align="center">CPO</th>
            <th align="center">partial instances</th>
        </tr>
        <tr>
            <td align="center">Tracklets(ID/Images)</td>
            <td align="center">89</td>
            <td align="center">283</td>
            <td align="center">90(randomly)</td>
        </tr>
        <tr>
            <td align="center">Fraction()</td>
            <td align="center">23.9%</td>
            <td align="center">76.1%</td>
            <td align="center">25%</td>
        </tr>
    </table>
</body>
</html>



<table border="1" style="width: 100%; text-align: center;">
    <caption>Table 5 Comparison with State-of-the-art on Mars and DukeMTMC-VideoReID</caption>  
    <tr>
    <td rowspan="2">Methods</th>
    <th colspan="3">Mars</th>
    <th colspan="3">DukeMTMC-VideoRelD</th>
    <th colspan="3">iLIDS-VID</th>
  </tr>
  <tr>
    <th>MAP</th>
    <th>Top1</th>
    <th>Top5</th>
    <th>MAP</th>
    <th>Top1</th>
    <th>Top5</th>
    <th>MAP</th>
    <th>Top1</th>
    <th>Top5</th>
  </tr>
  <tr>
    <td>VRSTC(CVPR2019)</td>
    <td>82.3</td>
    <td>88.5</td>
    <td>96.5</td>
    <td>93.5</td>
    <td>95.0</td>
    <td>99.1</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>CTL(CVPR2021)</td>
    <td>83.7</td>
    <td>90.0</td>
    <td>96.4</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>STGCN(CVPR2020)</td>
    <td>83.7</td>
    <td>89.95</td>
    <td>95.7</td>
    <td>97.29</td>
    <td>99.3</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>BiCnet(CVPR2021)</td>
    <td>86.0</td>
    <td>90.2</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>SSN3DD(AAAI2021)</td>
    <td>86.2</td>
    <td>90.1</td>
    <td>96.6</td>
    <td>96.3</td>
    <td>96.8</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>GPNet(2023)</td>
    <td>85.1</td>
    <td>90.2</td>
    <td>96.8</td>
    <td>96.1</td>
    <td>96.3</td>
    <td>99.6</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>STMN(ICCV21)</td>
    <td>84.5</td>
    <td>90.5</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>PSTA(ICCV21)</td>
    <td>85.8</td>
    <td>91.5</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>91.5</td>
    <td>98.1</td>
    <td>-</td>
  </tr>
  <tr>
    <td>DIL(ICCV21)</td>
    <td>87.0</td>
    <td>90.8</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>92.0</td>
    <td>98.0</td>
    <td>-</td>
  </tr>
  <tr>
    <td>TMT(Arxiv21)</td>
    <td>85.8</td>
    <td>92.1</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>91.3</td>
    <td>98.6</td>
    <td>-</td>
  </tr>
  <tr>
    <td>CAVIT(ECCV22)</td>
    <td>87.2</td>
    <td>90.8</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>93.3-</td>
    <td>98.0</td>
    <td>-</td>
  </tr>
  <tr>
    <td>SINet(CVPR22)</td>
    <td>86.2</td>
    <td>91.0</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>92.5</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>MFA(TIP22)</td>
    <td>85.0</td>
    <td>90.4</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>93.3</td>
    <td>98.7</td>
    <td>-</td>
  </tr>
  <tr>
    <td>DCCT(TNNLS23)</td>
    <td>87.5</td>
    <td>92.3</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>91.7</td>
    <td>98.6</td>
    <td>-</td>
  </tr>
  <tr>
    <td>LSTRL(IGIG23)</td>
    <td>86.8</td>
    <td>91.6</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>92.2</td>
    <td>98.6</td>
    <td>-</td>
  </tr>
  <tr>
    <td>HGSD</td>
    <td>88.653</td>
    <td>93.734</td>
    <td>98.177</td>
    <td>97.587</td>
    <td>98.296</td>
    <td>99.732</td>
    <td>94.256</td>
    <td>98.637</td>
    <td>99.153</td>
  </tr>
  <tr>
    <td>RECnet(TPAMI21)[16]</td>
    <td>86.3</td>
    <td>90.7</td>
    <td>-</td>
    <td>97.0</td>
    <td>97.6</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>SANet(TCSVT21)[17]</td>
    <td>86.0</td>
    <td>91.2</td>
    <td>-</td>
    <td>96.7</td>
    <td>97.7</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>IRNet(TCSVT23)[18]</td>
    <td>87.0</td>
    <td>92.5</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>92.9</td>
    <td>-</td>
  </tr>
  <tr>
     <tr>
    <td>HMN(TCSVT22)[19]</td>
    <td>82.6</td>
    <td>88.5</td>
    <td>96.2</td>
    <td>95.1</td>
    <td>96.3</td>
    <td>99.2</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>SGMN[20]</td>
    <td>85.38</td>
    <td>90.76</td>
    <td>96.90</td>
    <td>96.30</td>
    <td>96.87</td>
    <td>99.60</td>
    <td>-</td>
    <td>88.68</td>
    <td>96.67</td>
  </tr>
  <tr>
    <td>SDCL[CVPR23][21]</td>
    <td>86.5</td>
    <td>91.1</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>93.2</td>
    <td>92.7</td>
    <td>-</td>
  </tr>
  <tr>
    <td>STFE[TTM24][22]</td>
    <td>91.5</td>
    <td>95.5</td>
    <td>97.9</td>
    <td>97.0</td>
    <td>97.6</td>
    <td>99.7</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>MSFANet[KBS24][23]</td>
    <td>87.0</td>
    <td>91.4</td>
    <td>97.0</td>
    <td>96.8</td>
    <td>97.3</td>
    <td>99.6</td>
    <td>-</td>
    <td>93.4</td>
    <td>99.3</td>
  </tr>
  <tr>
    <td>SFGN[KBS22][24]</td>
    <td>85.6</td>
    <td>89.9</td>
    <td>96.4</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>89.7</td>
    <td>97.0</td>
  </tr>
  <tr>
    <td>MMA-GGA[TCSVT22][25]</td>
    <td>85.4</td>
    <td>91.0</td>
    <td>97.0</td>
    <td>96.2</td>
    <td>97.3</td>
    <td>99.6</td>
    <td>-</td>
    <td>87.7</td>
    <td>98.7</td>
  </tr>
  <tr>
    <td>DSANet[WACV23][26]</td>
    <td>86.6</td>
    <td>91.1</td>
    <td>-</td>
    <td>96.6</td>
    <td>97.2</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>HASI[TCSVT23][27]</td>
    <td>87.5</td>
    <td>91.4</td>
    <td>97.4</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>96.1</td>
    <td>93.3</td>
    <td>99.6</td>
  </tr>
  <tr>
    <td>SGWCNN[PR22][28]</td>
    <td>85.7</td>
    <td>90.0</td>
    <td>97.0</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>87.8</td>
    <td>96.0</td>
  </tr>
  <tr>
    <td>PiT[IT22][29]</td>
    <td>86.8</td>
    <td>90.2</td>
    <td>97.2</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>92.1</td>
    <td>98.9</td>
  </tr>
  <tr>
    <td>CSANet[TOMM23][30]</td>
    <td>86.5</td>
    <td>90.4</td>
    <td>96.7</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>90.0</td>
    <td>98.3</td>
  </tr>
  <tr>
    <td>Graph-Trans[BBE24][31]</td>
    <td>86.4</td>
    <td>92.5</td>
    <td>97.5</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>93.8</td>
    <td>98.6</td>
  </tr>
  <tr>
    <td>TE-CLIP[AAAI24][32]</td>
    <td>89.4</td>
    <td>93.0</td>
    <td>97.9</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>94.5</td>
    <td>99.5</td>
  </tr>
</table>


<table border="1" style="width: 100%; text-align: center;">
  <caption>Table 6 Comparison with State-of-the-art on Occluded Dataset</caption>
  <thead>
    <tr>
      <th rowspan="2">Occlusion</th>
      <th rowspan="2">methods</th>
      <th colspan="4">Occluded-Duke</th>
      <th colspan="4">Occluded-Duke-VideoReID</th>
      <th colspan="4">UESTC-SD</th>
    </tr>
    <tr>
      <th>mAP</th>
      <th>Top-1</th>
      <th>Top-5</th>
      <th>Top-10</th>
      <th>mAP</th>
      <th>Top-1</th>
      <th>Top-5</th>
      <th>Top-10</th>
      <th>mAP</th>
      <th>Top-1</th>
      <th>Top-5</th>
      <th>Top-10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5">APO</td>
      <td>VRSTC</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>76.7</td>
      <td>76.9</td>
      <td>90.3</td>
      <td>94.2</td>
      <td>42.648</td>
      <td>38.367</td>
      <td>49.502</td>
      <td>71.233</td>
    </tr>
    <tr>
      <td>RFCnet</td>
      <td>54.5</td>
      <td>63.9</td>
      <td>77.6</td>
      <td>82.1</td>
      <td>92.0</td>
      <td>93.0</td>
      <td>98.6</td>
      <td>99.1</td>
      <td>67.736</td>
      <td>65.601</td>
      <td>73.846</td>
      <td>81.343</td>
    </tr>
    <tr>
      <td>EcReID</td>
      <td>52.7</td>
      <td>64.8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>STGCN</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>62.378</td>
      <td>60.937</td>
      <td>67.482</td>
      <td>75.196</td>
    </tr>
    <tr>
      <td>HGSD</td>
      <td>56.371</td>
      <td>66.755</td>
      <td>79.078</td>
      <td>82.518</td>
      <td>92.308</td>
      <td>94.017</td>
      <td>98.725</td>
      <td>99.742</td>
      <td>80.286</td>
      <td>81.749</td>
      <td>85.062</td>
      <td>80.286</td>
    </tr>
    <tr>
      <td rowspan="3">CPO</td>
      <td>(FED)</td>
      <td>56.4</td>
      <td>68.1</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>(ref)</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>HGSD</td>
      <td>56.371</td>
      <td>66.755</td>
      <td>79.078</td>
      <td>82.518</td>
      <td>92.308</td>
      <td>94.017</td>
      <td>98.725</td>
      <td>99.742</td>
      <td>80.286</td>
      <td>81.749</td>
      <td>85.062</td>
      <td>87.357</td>
    </tr>
    <tr>
      <td rowspan="3">Partial instances</td>
      <td>ref</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ref</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>HGSD</td>
      <td>56.371</td>
      <td>66.755</td>
      <td>79.078</td>
      <td>82.518</td>
      <td>92.308</td>
      <td>94.017</td>
      <td>98.725</td>
      <td>99.742</td>
      <td>80.286</td>
      <td>81.749</td>
      <td>85.062</td>
      <td>87.357</td>
    </tr>
  </tbody>
</table>


<table border="1" style="width: 100%; text-align: center;">
  <caption>Table 7 Ablation Study on Occluded-Duck-Video and UESTC-SD</caption>
  <thead>
    </tr>
    <tr>
      <th></th>
      <th colspan="3" style="text-align: center;">Occluded-Duke-VideoRe-ID</th>
      <th colspan="3" style="text-align: center;">UESTC-SD</th>
    </tr>
    <tr>
      <th style="text-align: center;"></th>
      <th style="text-align: center;">mAP</th>
      <th style="text-align: center;">Top-1</th>
      <th style="text-align: center;">Top-5</th>
      <th style="text-align: center;">mAP</th>
      <th style="text-align: center;">Top-1</th>
      <th style="text-align: center;">Top-5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="7">part1</td>
    </tr>
    <tr>
      <td>Baseline</td>
      <td>73.528</td>
      <td>75.072</td>
      <td>79.427</td>
      <td>62.682</td>
      <td>63.535</td>
      <td>68.903</td>
    </tr>
    <tr>
      <td>HGSD-E</td>
      <td>80.159</td>
      <td>80.743</td>
      <td>84.228</td>
      <td>71.595</td>
      <td>70.318</td>
      <td>76.192</td>
    </tr>
    <tr>
      <td>HGSD-N</td>
      <td>84.318</td>
      <td>86.267</td>
      <td>91.072</td>
      <td>74.378</td>
      <td>74.501</td>
      <td>79.361</td>
    </tr>
    <tr>
      <td>HGSD-EN</td>
      <td>92.308</td>
      <td>94.017</td>
      <td>98.725</td>
      <td>80.286</td>
      <td>81.749</td>
      <td>85.062</td>
    </tr>
    <tr>
      <td colspan="7">part2</td>
    </tr>
    <tr>
      <td>HGSD-wo-Ltri</td>
      <td>85.392</td>
      <td>86.767</td>
      <td>92.158</td>
      <td>76.677</td>
      <td>75.832</td>
      <td>82.193</td>
    </tr>
    <tr>
      <td>HGSD-wo-Ls</td>
      <td>87.019</td>
      <td>88.462</td>
      <td>93.407</td>
      <td>77.132</td>
      <td>77.344</td>
      <td>83.081</td>
    </tr>
    <tr>
      <td>HGSD-wo-Lp</td>
      <td>89.517</td>
      <td>91.032</td>
      <td>95.372</td>
      <td>79.352</td>
      <td>78.306</td>
      <td>81.255</td>
    </tr>
    <tr>
      <td>HGSD-wi-K1</td>
      <td>92.308</td>
      <td>94.017</td>
      <td>98.725</td>
      <td>80.286</td>
      <td>81.749</td>
      <td>85.062</td>
    </tr>
    <tr>
      <td>HGSD-wi-K2</td>
      <td>91.361</td>
      <td>92.507</td>
      <td>95.336</td>
      <td>78.732</td>
      <td>81.067</td>
      <td>83.446</td>
    </tr>
    <tr>
      <td colspan="7">part3</td>
    </tr>
    <tr>
      <td>HGSD-E</td>
      <td>80.159</td>
      <td>80.743</td>
      <td>84.228</td>
      <td>71.595</td>
      <td>70.318</td>
      <td>76.192</td>
    </tr>
    <tr>
      <td>HGSD-N</td>
      <td>84.318</td>
      <td>86.267</td>
      <td>91.072</td>
      <td>74.378</td>
      <td>74.501</td>
      <td>79.361</td>
    </tr>
    <tr>
      <td>HGSD-E-N</td>
      <td>90.167</td>
      <td>92.381</td>
      <td>96.722</td>
      <td>79.672</td>
      <td>78.561</td>
      <td>84.306</td>
    </tr>
    <tr>
      <td>HGSD-N-E</td>
      <td>92.308</td>
      <td>94.017</td>
      <td>98.725</td>
      <td>80.286</td>
      <td>81.749</td>
      <td>85.062</td>
    </tr>
    <tr>
      <td colspan="7">part4</td>
    </tr>
    <tr>
      <td>HGSD-16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>HGSD-32</td>
      <td>91.187</td>
      <td>92.548</td>
      <td>98.913</td>
      <td>79.239</td>
      <td>78.481</td>
      <td>83.352</td>
    </tr>
    <tr>
      <td>HGSD-64</td>
      <td>92.308</td>
      <td>94.017</td>
      <td>98.725</td>
      <td>80.286</td>
      <td>81.749</td>
      <td>85.062</td>
    </tr>
  </tbody>
</table>

# Reid person search公开数据集

数据库查询网址：

【1】https://paperswithcode.com/

【2】https://www.kaggle.com/datasets

【3】[数据集 / HyperAI超神经](https://hyper.ai/cn/datasets)

【4】https://aistudio.baidu.com/datasetdetail

## ReID概述

行人重识别是指利用计算机视觉技术，判断在不同时间段、不同监控下出现的行人图像是否属于同一人员的技术。行人重识别是最近几年在视频分析领域下热门的研究领域，可以看做是人脸识别应用的拓展。现在大街上的监控较多，由于设备质量、成像光线、成像角度、以及成像距离的因素，在监控视频中得到的人的特征往往是不清晰的，人脸的分辨率是不足以做人脸识别的，所以提出了根据行人的穿着、体态、发型等信息认知行人的reid。可与行人检测、行人跟踪技术相结合，可以弥补目前监控摄像头的视觉局限性，可以广泛应用于监控、安防等领域。
研究特点：

1. 研究对象：
   研究的对象是人的整个特征，不以人脸识别作为手段，包括衣着、体态、发型、姿态等等。
2. 跨摄像头：
   关注跨摄像头上的行人再识别问题。
3. 人脸的关系：
   可以作为人脸识别技术的延伸，并相互作用，应用于更多场景。
4. 具有时空连续性以及短时性。

## ReID相关数据集

### 图像数据集（Image DataSet）

| DataSet         | 下载链接                                                     | 介绍                                                         |
| :-------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| VIPeR           | [ Computer Vision Lab](https://vision.soe.ucsc.edu/user/login?destination=node%2F178) | VIPeR数据集包括632个行人的1264张图像，每个行人有两张图像，采集自摄像头a和摄像头b. 每张图像都调整为了128x48的大小。该数据集的特点为视角和光照的多样性。 |
| iLIDS           | [JDAI-CV/Partial-Person-ReID](https://github.com/JDAI-CV/Partial-Person-ReID) |                                                              |
| GRID            | [QMUL underGround Re-IDentification (GRID) Dataset](https://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html) | QMUL地下再识别(GRID)数据集包含250个行人图像对。每一对都包含两张从不同视角看到的同一个体的图像。所有的图像都是从安装在一个繁忙的地铁站的8个不相交的摄像头视图中捕捉到的。旁边的图显示了该站的每个相机视图的快照和数据集中的样本图像。由于姿势的变化，颜色，灯光的变化，数据集是具有挑战性的;以及低空间分辨率造成的图像质量差。 |
| CUHK01          | [CUHK Re-ID](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) |                                                              |
| CUHK02          | [CUHK Re-ID](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) | CUHK02 是一个用于行人重识别的数据集。该数据集共有 1,816 幅图像，从两个不同的摄像视角，拍摄了共 1,816 位行人，每位行人的每个摄像角度各有两张图像样本。 |
| CUHK03          | [CUHK Re-ID](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) |                                                              |
| Market-1501     | [market_1501](https://www.kaggle.com/datasets/pengcw1/market-1501/data) | Market-1501 数据集在清华大学校园中采集，夏天拍摄，在 2015 年构建并公开。它包括由6个摄像头（其中5个高清摄像头和1个低清摄像头）拍摄到的 1501 个行人、32668 个检测到的行人矩形框。每个行人至少由2个摄像头捕获到，并且在一个摄像头中可能具有多张图像。训练集有 751 人，包含 12,936 张图像，平均每个人有 17.2 张训练数据；测试集有 750 人，包含 19,732 张图像，平均每个人有 26.3 张测试数据。3368 张查询图像的行人检测矩形框是人工绘制的，而 gallery 中的行人检测矩形框则是使用DPM检测器检测得到的。该数据集提供的固定数量的训练集和测试集均可以在single-shot或multi-shot测试设置下使用 |
| DukeMTMC        | [dukemtmc-reid](https://www.kaggle.com/datasets/igorkrashenyi/dukemtmc-reid?select=README.md) |                                                              |
| Airport         | northeastern网站删除                                         | Airport 是一个用于行人重新识别的数据集，包含来自 6 台摄像机的 39,902 张图像和 9,651 个身份。每个身份平均拥有 3.13 张图像。每张图像的分辨率为 128×64 像素。 |
| MSMT17          | [Welcome to Shiliang Zhang's Homepage](https://www.pkuvmc.com/dataset.html) | MSMT17 是一个多场景、多时间的行人重识别数据集。该数据集由 180 小时的视频组成，由 12 个室外摄像头、3 个室内摄像头和 12 个时间段捕获。视频时长，呈现复杂的光照变化，包含大量带注释的身份，即 4101 个身份和 126441 个边界框。 |
| ETH             | https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz | 与其他数据集从多台相机收集图像不同，ETHZ从一个移动的相机收集图像。虽然视点方差比较小，但它确实有相当大的光照方差、尺度方差和遮挡。 |
| SYSU-30k        | [wanggrun/SYSU-30k: SYSU-30k Dataset of "Weakly Supervised Person Re-ID: Differentiable Graphical Learning and A New Benchmark" https://arxiv.org/abs/1904.03845](https://github.com/wanggrun/SYSU-30k) | SYSU-30k 数据集包含 30,000 个行人身份类别，约是 CUHK03 数据集 和 [Market-1501 数据集](https://orion.hyper.ai/datasets/16157)的 20 倍。如果一个行人身份类别相当于一个物体类别的话，则 SYSU-30k 相当于[ ImageNet 数据集](https://orion.hyper.ai/datasets/4889)的 30 倍。该数据集总共包含 29,606,918 张图像。 SYSU-30k 数据集不仅仅为弱监督 ReID 问题提供一个评测平台，更是一个符合现实场景挑战性的测试集。 |
| PKU-Reid        | [charliememory/PKU-Reid-Dataset: PKU-Reid dataset](https://github.com/charliememory/PKU-Reid-Dataset) | 此数据集共涉及 114 个不同的行人，包含 1,824 张图像，分别用两个不相交的摄像机从 8 个方向拍摄得到。对于每个人，在一个相机视图下从八个不同方向捕获了八张图像，尺寸标准为 128×48 像素。该数据集也被随机分为两部分：57 个人用于训练，57 个人用于测试。 |
| P-DukeMTMC-reID | [tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets: Release the datasets in the paper（ ICME 2018 ） ,Occluded Person Re-identification.](https://github.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets) | P-DukeMTMC-reID 是一个遮挡行人数据集。该数据集是从 DukeMTMC-reID dataset 中人工挑选出来的。 P-DukeMTMC-reID 数据集中，12,927 张图像（665 个身份）用于训练，2,163 张图像（634 个标识）用于查询，图库集包含 9,053 个图像。 |
| Occluded REID   | [tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets: Release the datasets in the paper（ ICME 2018 ） ,Occluded Person Re-identification.](https://github.com/tinajia2012/ICME2018_Occluded-Person-Reidentification_datasets) | Occluded REID 是一个有遮挡的行人图像数据集，由手机相机拍摄，主要用于有遮挡的行人重识别研究。该数据集包含了 200 位被遮挡都行人，图像共计 2,000 张，每位行人有 5 张全身图像，以及 5 张从不同角度被遮挡的图像。 |

### 视频数据集（Vedio DataSet）

| DataSet   | 链接                                                         | 介绍相关页面                                                 |
| :-------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| PRID-2011 | [prid2011(video reid)_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/datasetdetail/120155) | PRID数据集有摄像机A的385条轨迹和摄像机b的749条轨迹，其中只有200人出现在两个摄像机中。该数据集还有一个单镜头版本，由随机选择的快照组成。有些轨迹没有很好地同步，这意味着人可能会在连续的帧之间“跳跃”。 |
| iLIDS-VID | [Xiatian Zhu](https://xiatian-zhu.github.io/downloads_qmul_iLIDS-VID_ReID_dataset.html) | 该数据集是根据 [i-LIDS 多摄像头跟踪场景 （MCTS） 数据集](https://www.gov.uk/imagery-library-for-intelligent-detection-systems)的两个不重叠摄像头视图中观察到的行人创建的，该数据集是在多摄像头 CCTV 网络下的机场到达大厅捕获的。 它由 300 个不同个体的 600 个图像序列组成，每个个体的一对图像序列来自两个相机视图。 每个图像序列的长度可变，范围从 23 到 192 个图像帧，平均数量为 73 个。 iLIDS-VID 数据集非常具有挑战性，因为人物之间的服装相似性、相机视图之间的照明和视点变化， 杂乱的背景和随机遮挡。 为了便于在此数据集上评估基于单次的人物重新识别方法， 我们还通过从每个人的图像序列中随机选择一张图像来提供基于静态图像的版本。*提供了基准培训/测试人员拆分，以便在文献中不同最先进的方法之间进行公平比较。* |
| MARS      | [MARS_ReID_Dataset_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/datasetdetail/76843) | 在Market501上进行扩充的行人重识别数据集。                    |
| Duke-MTMC | [Yu-Wu/DukeMTMC-VideoReID: Instructions and baseline code for the DukeMTMC-VideoReID dataset](https://github.com/Yu-Wu/DukeMTMC-VideoReID) | DukeMTMC 数据集采集自 Duke 大学的 8 个摄像头，数据集以视频形式存储，具有手动标注的行人边界框。DukeMTMC-reID 数据集从 DukeMTMC 数据集的视频中，每 120 帧采集一张图像构成 DukeMTMC-reID 数据集。 |
| LPW       | [Labeled Pedestrian in the Wild](https://liuyu.us/dataset/lpw/index.html) | LPW 全称 Labeled Pedestrian in the Wild，是一个行人检测数据集。该数据集包含三个不同的场景，共 2,731 名行人，其中每一个行人图片由 2 到 4 台摄像机拍摄。 LPW 的显着特征是：包含 7,694 个 tracklets，超过 590,000 个图像。LPW 与现有的数据集有三个重要区别：大规模且干净度高、自动检测边界框、更多拥挤的场景和更大的年龄跨度。该数据集提供了一个更真实、更具挑战性的基准，这有助于进一步探索更强大的算法。 |
| LS-VID    | [Welcome to Shiliang Zhang's Homepage](https://www.pkuvmc.com/dataset/dataset.html) | **iLIDS-VID** 数据集是一个人员再识别数据集，涉及在公共开放空间的两个不相交的相机视图中观察的 300 名不同的行人。它由 300 个不同个体的 600 个图像序列组成，每个个体的一对图像序列来自两个相机视图。每个图像序列的长度可变，范围从 23 到 192 个图像帧，平均数量为 73 个。iLIDS-VID 数据集非常具有挑战性，因为人物之间的服装相似性、摄像机视图之间的照明和视点变化、杂乱的背景和随机遮挡。 |

