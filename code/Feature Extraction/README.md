The purpose of features extraction and selection is to reduce the original data set by measuring certain properties that distinguish one input pattern from another pattern. The extracted feature should provide the characteristics of the input type to the classifier by considering the description of relevant properties of the image into a feature space.
In our experiments we have used wavelet as feature extraction as it has been widely used. wavelet as 14 family which are

1. Haar (haar)

2. Daubechies (db)

3. Symlets (sym)

4. Coiflets (coif)

5. Biorthogonal (bior)

6. Reverse biorthogonal (rbio)

7. “Discrete” FIR approximation of Meyer wavelet (dmey) 8. Gaussian wavelets (gaus)

9. Mexican hat wavelet (mexh)

10. Morlet wavelet (morl)

11. Complex Gaussian wavelets (cgau) 12. Shannon wavelets (shan)

13. Frequency B-Spline wavelets (fbsp) 14. Complex Morlet wavelets (cmor)

For some families we use different vanishing moments

• Daubechies (db)
’db1’, ’db2’, ’db3’, ’db4’, ’db5’, ’db6’, ’db7’, ’db8’, ’db9’, ’db10’, ’db11’, ’db12’, ’db13’, ’db14’, ’db15’, ’db16’, ’db17’, ’db18’, ’db19’, ’db20’, ’db21’, ’db22’, ’db23’, ’db24’, ’db25’, ’db26’, ’db27’, ’db28’, ’db29’, ’db30’, ’db31’, ’db32’, ’db33’, ’db34’, ’db35’, ’db36’, ’db37’, ’db38’

• Symlets (sym)
’sym2’, ’sym3’, ’sym4’, ’sym5’, ’sym6’, ’sym7’, ’sym8’, ’sym9’, ’sym10’, ’sym11’, ’sym12’,
’sym13’, ’sym14’, ’sym15’, ’sym16’, ’sym17’, ’sym18’, ’sym19’, ’sym20’

• Coiflets (coif)
’coif1’, ’coif2’, ’coif3’, ’coif4’, ’coif5’, ’coif6’, ’coif7’, ’coif8’, ’coif9’, ’coif10’, ’coif11’, ’coif12’,
’coif13’, ’coif14’, ’coif15’, ’coif16’, ’coif17’

• Biorthogonal (bior)
’bior1.1’, ’bior1.3’, ’bior1.5’, ’bior2.2’, ’bior2.4’, ’bior2.6’, ’bior2.8’, ’bior3.1’, ’bior3.3’, ’bior3.5’,
’bior3.7’, ’bior3.9’, ’bior4.4’, ’bior5.5’, ’bior6.8’

• Reverse biorthogonal (rbio)
’rbio1.1’, ’rbio1.3’, ’rbio1.5’, ’rbio2.2’, ’rbio2.4’, ’rbio2.6’, ’rbio2.8’, ’rbio3.1’, ’rbio3.3’, ’rbio3.5’,
’rbio3.7’, ’rbio3.9’, ’rbio4.4’, ’rbio5.5’, ’rbio6.8’

• Gaussian wavelets (gaus)
’gaus1’, ’gaus2’, ’gaus3’, ’gaus4’, ’gaus5’, ’gaus6’, ’gaus7’, ’gaus8’
• Complex Gaussian wavelets (cgau) ’cgau1’, ’cgau2’, ’cgau3’, ’cgau4’, ’cgau5’, ’cgau6’, ’cgau7’, ’cgau8’
