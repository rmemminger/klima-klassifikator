# klima-klassifikator
Dieses Repository beinhaltet einen trainierten Textklassifikator, mit dem deutsche politische Texte zum Thema Klimawandel gefiltert werden können. Der Klassifikator wurde auf Absätzen von Texten der Partei Alternative für Deutschland trainiert, ist jedoch allgemein auf kurze politische Texte anwendbar. Es handelt sich um ein XGBoost Modell, welches eine binäre Textklassifikation vornimmt. Eingespeiste Texte werden entweder mit dem Label 0 (nicht themenrelevant) oder 1 (themenrelevant) versehen. Genaueres zur Erstellung, sowie Auswertung des Modells ist im [Abstract](Abstract.md) nachzulesen.

## Anwendung des Klassifikators
Zur vereinfachten Anwendung des Klassifikators zur Textfilterung ist ein Jupyter Notebook bereitgestellt. 

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
