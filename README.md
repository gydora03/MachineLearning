# Gépi tanulás a WebsitePhishing adatállománnyra

Az általam választott adatállomány neve Website Phishing, melyet a Kaggle-ről töltöttem le, de elérhető a UCI Machine Learning Repository-n is. 
A Website Phishing adatállomány 1353 rekordot és 10 attribútumot tartalmaz. <br/> <br/> 
Az adathalász problémát létfontosságú kérdésnek tekintik az iparban, különösen az e-banki és az e-kereskedelemben, figyelembe véve a fizetésekkel járó online tranzakciók számát. 
Különböző szolgáltatásokat azonosítottak a törvényes és adathalász webhelyekkel kapcsolatban és 1353 különböző webhelyet gyűjtöttek össze különböző forrásokból. 
Ebből 548 legális, 702 adathalász és 103 gyanús (egyben legális és adathalász) webhely. A jelöléseket tekintve pedig: 1 = legális, 0 = gyanús és -1 = adathalász. <br/> <br/> 
Az adatállomány beimportálása, mint felügyelt, mint nem felügyelt tanítás során, kétféleképpen is meg van valósítva: először a Pandas, majd a Numpy segítségével. 
Mindkét esetben az adatállomány betöltése egy URL-ről történik, mely URL egyik saját GitHub Repositorymra mutat, ahol az adatállomány .csv és .data formátumban is megtalálható. <br/>
A felügyelt tanítás során 4 tanult módszert alkalmaztam: Lineáris regresszió, Logisztikus Regresszió, Perceptron és Neurális háló. 
A lineáris regresszió során a modell illesztésének a jósága 0.63, a logisztikus regresszióé 0.83, a preceptroné 0.82 és a neurális hálóé pedig 0.82. 
Ezek alapján azt mondhatjuk, hogy a lejobb illesztést (scoret) a logisztikus regresszió estén kaptuk. 
Mindegyik módszer szemléltetve van grafikus ábrákkal, továbbá a kiértékelések során ROC görbékkel és tévesztési mátrixokkal is találkozunk. <br/> <br/> 
A nemfelügyelt tanítás során a Kmean klaszterezést alkalmaztam. Az adatállomány beimportálása után a felhasználótól bekérem a klaszterek számát, 
melyekkel a későbbiekben a program dolgozni fog. Ezután rögtön el is készül egy klaszterezés a megadott klaszterszámmal, majd az illetsztett modell jóságának 
a meghatározásának céljából kiszámolódik egy Davies-Bouldin index. A klaszterezés, egy dimenzió csökkentést követően, grafikus ábrák segítségével is vizualizálva lesz, 
mint a főkomponensek, mint a klaszter-középpontoktól való távolságok terébe. Végül jön az optimális klaszterszám megkeresése, mely során egy SSE (négyzetösszeg) 
és egy Davies-Bouldin index ábrázolása is megtörténik. A Davies-Bouldin görbe segít nekünk meghatározni az optimális klaszterszámot, ez a szám a görbe lokális minimumának 
a helye lesz. Erre a számra kipróbálva a Davies-Bouldin index 1.496 lesz. <br/> <br/> 
