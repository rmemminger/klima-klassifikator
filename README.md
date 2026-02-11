# klima-klassifikator
Dieses Repository beinhaltet einen trainierten Textklassifikator, mit dem deutsche politische Texte zum Thema Klimawandel gefiltert werden können. Der Klassifikator wurde auf Absätzen von Texten der Partei Alternative für Deutschland trainiert, ist jedoch allgemein auf kurze politische Texte anwendbar. Es handelt sich um ein XGBoost Modell, welches eine binäre Textklassifikation vornimmt. Eingespeiste Texte werden entweder mit dem Label 0 (nicht themenrelevant) oder 1 (themenrelevant) versehen. Genaueres zur Erstellung, sowie Auswertung des Modells ist im [Abstract](Abstract.md) nachzulesen.

## Anwendung des Klassifikators
Der Klassifikator kann mithilfe des Skripts `klassifikator.py` einfach über die Command Line ausgeführt werden. Hierbei werden folgende Argumente genutzt:

1. `--csv [Pfad zur Datei]` __oder__ `--txt [Pfad zum Ordner]`

    Dies gibt an, ob Texte aus einer gesammelten csv Datei oder einem Ordner an txt Dateien ausgelesen werden sollen. Der angegebene Dateienpfad muss im Ordner liegen, in dem sich das Skript befindet.
2. `--embed` __oder__ `--load-embed [Pfad zur Datei]` 

    Beim ersten Ausführen müssen Text Embeddings erstellt werden. Mit dem Argument --embed wird dies durchgeführt und die Embeddings werden gespeichert. Um die Rechenlast zu verringern, können bei erneuten Ausführungen die bereits erstellten Embeddings geladen werden. Daher muss bei --load-embed auch der Pfad zur Embedding Datei (automatisch: embeddings.npy) angegeben werden.

Beispiel CLI:

    klassifikator.py --txt texts --load-embed embeddings.npy

Zur vereinfachten Anwendung des Klassifikators zur Textfilterung ist außerdem ein Jupyter Notebook bereitgestellt. 

### Voraussetzungen
**Setup:** Die Anwendung des Klassifikators benötigt eine Python 3 Umgebung. Für ein reibungsloses Setup empfielt sich die Installation der benötigten Packages aus `requirements.txt` in ein virtuelles environment.
Wir empfehlen den Klassifikator nur bei Verfügbarkeit einer GPU anzuwenden, da der Embedding Prozess ressourcenintensiv ist.

**Daten:** 
- Input Texte sollten Absatzlänge haben und idealerweise eine Länge von 512 Tokens nicht überschreiten. Zwar ist eine Klassifikation auch bei Überlänge möglich, jedoch schneidet das Embedding Modell den Text nach der Maximallänge ab, wodurch nur die ersten 512 Tokens bei der Klassifikation berücksichtigt werden können. Da der Klassifikator außerdem auf kürzeren Texten (<500 Tokens) trainiert wurde, empfielt sich die Anwendung auf vergleichbare Textlängen. 
- Texte zur Klassifikation sollten im pandas DataFrame Format mit einer Spalte 'text' bereitgestellt werden.

### Ablauf

1. Stichworte: Als Vorbereitung für die Klassifikation werden Häufigkeiten bestimmter Stichworte in jedem Absatz gezählt und gespeichert.
2. Embeddings: Text werden mit dem Modell `bert-base-german-cased` in Vektorformat umgewandelt. Diese werden gespeichert und können bei wiederholter Klassifikation der selben Texte abgerufen werden, ohne den Schritt wiederholen zu müssen.
3. Erstellen der Input Matrix: Die vektorisierten Texte werden mit den Stichwortzahlen in einer Matrix zusammengefügt, wobei jede Zeile einem Text entspricht.
4. Klassifikation: Das Modell wird für Inferenz geladen und die Matrix zur Klassifikation eingespeist. Das Output beinhaltet die Texte im ursprünglichen Format, sowie die Stichwortzahlen und das Ergebnis der Klassifikation in der Spalte 'pred_y'.

## Verwendung und Lizenz
Bei Anwendung des Klassifikators zitieren Sie bitte dieses Repository und die Autoren: Memminger, Benndorf und Stede (2025). Der Klassifikator ist lizensiert unter BY-NC-SA 4.0 und darf nicht zu kommerziellen Zwecken verwendet werden.
