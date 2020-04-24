# Werkboek Analytical Computing (V1-AC)

Op deze Git-repo vind je het werkboek van Analytical Computing. Maak een eigen fork, clone deze op je computer en je kan aan de slag.

Succes!
Brian

## Handige tips:
- Maak een fork van deze repository waar je eigen werk komt te staan.
- Maak een SSH key aan (onder Linux kan dit met `ssh-keygen`, onder Windows zit dit ergens in Git For Windows) en upload je public key naar GitLab voordat je een lokale clone maakt; je hoeft dan niet iedere keer je credentials op te geven om (vanaf de PC/account waarop je de key gemaakt hebt) je code naar GitLab te pushen.
- Maak daarna een clone van je eigen repository op je eigen computer of laptop.
- Installeer Jupyter Notebook via Docker, of alternatief Anaconda. 
  - Anaconda is een package manager waarmee je Python-onderdelen makkelijk kan installeren. Je draait Jupyter vanuit een speciale omgeving.
  - In het bestand `docker-compose.yml` is een configuratie gegeven om alle verdere benodigdheden in een enkele Docker container te hebben. Dit is een soort mini virtueel systeem dat enkel Jupyter en de benodigdheden bevat. Je start Docker For Windows (of een terminal in Linux of Mac) en voert vanuit de `v1ac` map het commando `docker-compose up` uit om Jupyter te starten. De gebruikte Docker-image bevat alle benodigdheden voor de meeste data-science toepassingen.
- In beide gevallen kun je vervolgens in je browser naar `localhost:8888` of `127.0.0.1:8888` om in Jupyter te komen.
