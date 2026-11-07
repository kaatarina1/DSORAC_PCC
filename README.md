# Ovrednotenje predstavitve z Gaussovimi packami z uporabo različnih pristopov 
Oblaki točk so eden najpogostejših načinov digitalnega zajema tridimenzionalnega sveta: namesto zveznih površin geometrijo preprosto opišejo z množico točk v prostoru, zato so združljive s skoraj vsako tehnologijo zajema: terestričnim in letalskim skeniranjem LiDAR (angl. Light Detection and Ranging), fotogrametrijo ali globinskimi kamerami, pojavljajo pa se tudi kot povsem sintetičen produkt računalniških simulacij. Poleg položaja pogosto nosijo tudi informacijo o barvi in druge atribute, zaradi česar so posebej uporabne pri predstavitvi resničnih objektov in prizorov. Težava nastane, ko taki zajemi postanejo gosti, saj sodobni senzorji brez težav ustvarijo oblake z več milijoni ali celo milijardami točk, količina podatkov pa hitro preraste zmožnosti njihovega shranjevanja, prenosa in predvsem upodabljanja v realnem času.

Zato je kompresija oblakov točk že dolgo aktivno raziskovano področje. Industrijska standarda MPEG V‑PCC in G‑PCC ter novejši AVS PCC problem rešujejo s hierarhičnimi prostorskimi strukturami ali tako, da posamezne okvirje dinamičnega oblaka točk projicirajo na 2D slike ter dobljeno zaporedje stisnejo z obstoječimi video kodirniki. Metode globokega učenja pa kompresijo obravnavajo kot problem, naučen neposredno iz podatkov. Vsem tem pristopom je skupno, da uspešnost kompresije ocenjujejo izključno na podlagi popačenja geometrije, torej kako natančno dekodiran oblak sledi izvirnim koordinatam točk, neodvisno od tega, kako bo rekonstruirana scena sploh upodobljena.

Kot obetavna alternativa se je zato uveljavila metoda 3D Gaussovega packanja, ki sceno namesto z mnogokotniki predstavi z množico anizotropnih Gaussovih porazdelitev (gaussovk), pri čemer ima vsaka svojo lego, obliko, prosojnost in od pogleda odvisno barvo, kodirano s sferičnimi harmoniki. Takšna eksplicitna predstavitev omogoča izjemno hiter in hkrati kakovosten izris iz poljubnega pogleda. Prav zaradi kompaktnega zapisa in hitrega upodabljanja je Gaussovo packanje zanimivo ne le za sintezo novih pogledov, za kar je bila prvotno zasnovano, temveč tudi kot način kompresije oblakov točk, ki namesto geometrijske natančnosti optimizira neposredno kakovost upodobitve.

Pri postopku 3D Gaussovega packanja naletimo na oviro, saj zahteva fotografije scene s poznanimi položaji kamer, ki pri oblakih točk pogosto niso na voljo. Peščica sorodnih del, med njimi Point2Gaussian in GS-PCC, to težavo rešuje tako, da manjkajoče fotografije nadomesti s sintetičnimi pogledi, zajetimi neposredno iz oblaka točk. V tem delu predstavljena metoda sledi enaki ideji, a se od obstoječih pristopov loči v razporeditvi kamer in tehniki rekonstrukcije sintetičnih pogledov. Kamere namesto po dveh koncentričnih sferah, kjer so razporejene enakomerno po azimutu in naklonu, razporedimo s Fibonaccijevo porazdelitvijo po eni ali več polsferah nad sceno. Pri rekonstrukciji sintetičnih pogledov pa namesto vnaprej naučenega transformerja uporabimo lasten postopek DSORAC, ki vrzeli zapolnjuje na podlagi globine, razdalje in gostote točk s klasično interpolacijo, s čimer je neodvisen od tipa scene in ne zahteva grafične kartice specifičnega proizvajalca.
 
Za preverjanje splošnosti in robustnosti smo predlagani pristop ovrednotili na štirih vsebinsko in po načinu zajema zelo raznolikih podatkovnih zbirkah: aerolasersko skeniranih scenah Slovenije, ki zajemajo teren, stavbe in rastlinje; javno dostopnih objektih iz zbirke Tanks and Temples; sintetično ustvarjenem oblaku točk brez šuma pri zajemu; ter gostih, a neenakomerno vzorčenih terestričnih zajemih mesta Ljubljana, zajetih iz premikajočega se vozila. Te scene skupaj pokrivajo razpon od redkih do gostih oblakov, od majhnih izoliranih objektov do obsežnih urbanih območij ter od sintetičnih do realnih podatkov.

Za vsako sceno smo ovrednotili štiri različne tehnike upodabljanja sintetičnih učnih slik: upodobitev s krogi, obrnjenimi proti kameri, s krogi in izotropnimi 2D gaussovkami, orientiranimi glede na normale točk, ter z našim pristopom DSORAC. Na dobljenih predstavitvah z gaussovkami smo primerjali kompresijsko razmerje, kvantitativne metrike kakovosti slike ter izgled upodobitev. Izvedli smo tudi študijo vpliva posameznih delov cevovoda, v kateri smo poleg tehnike izrisovanja spreminjali velikost inicializacijskega vzorca oblaka točk, uporabljenega pri optimizaciji gaussovk, in ovrednotili njegov vpliv na kakovost ter kompresijsko razmerje končne predstavitve. Nadalje pokažemo, da stopnje sferičnih harmonikov predstavljajo naraven mehanizem za prilagajanje ravni podrobnosti (angl. Level of Detail, LOD), ki ga primerjamo z uveljavljenimi strategijami LOD, ki temeljijo na zmanjševanju števila točk.

Glavni prispevki tega dela so:
- postopek za kompresijo oblakov točk v predstavitev z gaussovkami s postopkom 3D Guassovega packanja z uporabo sintetičnih upodobitev namesto fotografij;
- metoda upodabljanja DSORAC, ki pri rekonstrukciji sintetičnih pogledov upošteva globino, razdaljo in gostoto točk ter tako ustvari popolnejše in zanesljivejše učne slike brez potrebe po vnaprej naučenem modelu;
- študija vpliva tehnike upodabljanja na kakovost in kompresijsko razmerje optimiziranih gaussovk, v kateri sistematično primerjamo štiri tehnike upodabljanja na štirih raznolikih vrstah podatkov;
- študija vpliva velikosti inicializacijskega vzorca oblaka točk na kakovost in kompresijsko razmerje optimiziranih gaussovk;
- ovrednotenje sferičnih harmonikov kot mehanizma za prilagajanje ravni podrobnosti v primerjavi z zmanjševanjem števila gaussovk v sceni.

### Uporaba umetne inteligence
Pri implementaciji te zaključne naloge so bila uporabljena orogja umetne inteligence predvsem pri izdelavi dela uporabniškega vmesnika s kontrolami za upodabljanje (Control panel), strukturiranih logov pri izvedbi postopkov ter paralelizaciji postopka zajemanja slik. Prav tako smo jo včasih uporabljali za razhroščevanje med razvojem.
Refactor kode in strukturiranje repozitorija je bilo popolnoma narejeno z uporabo umetne intaligence.

## Namestitev

```bash
git clone --recursive https://github.com/kaatarina1/DSORAC.git
# ali, po navadnem kloniranju:
git submodule update --init --recursive

conda env create -f gaussian-splatting/environment.yml   # okolje gaussian_splatting za 3DGS, vključno z razširitvami CUDA
npm install                                              # odvisnosti za capture.js (deno task capture)
```

DXC (`dxcompiler.dll`, `dxil.dll`) gre v `tools/dxc/`.

## Struktura

```
pipeline.py                  # main(): celoten cevovod, zaženemo iz okolja cevovoda
gs_pipeline/                 # moduli, ki jih uvozi pipeline.py
gaussian_splatting_utils/    # skripte, ki tečejo v okolju gaussian_splatting (prek conda run)
experiments/                 # python -m experiments.<ime>
tests/                       # python -m tests.<ime>
configs/  scripts/
gaussian-splatting/          # podmodul, fork na veji pcc
```

Primer: `python pipeline.py data/pc/lego_normals.las --compression_ratio 10`.
Skripte v `experiments/` in `tests/` zaženemo iz korenskega direktorija kot module, npr. `python -m tests.test_optimization unit`.
