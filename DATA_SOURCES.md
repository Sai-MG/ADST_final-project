# Bengaluru Road Accident Data Sources Documentation

This document provides complete provenance, metadata, and methodological disclosures for all real-world datasets utilized in the **Bengaluru Road Accident Risk Prediction and Safety Recommendation System**.

---

## 1. Bengaluru Traffic Police (BTP) Station-Wise Total and Fatal Crashes (2018–2025)

* **Dataset Names**:
  * `btp_2025_station_wise.csv`
  * `btp_2024_station_wise.csv`
  * `btp_2023_station_wise.csv`
  * `btp_2020_2022_station_wise.csv`
  * `btp_2018_2020_station_wise.csv`
* **Source URL**: [https://data.opencity.in/dataset/bengaluru-road-crashes-data](https://data.opencity.in/dataset/bengaluru-road-crashes-data)
* **Publisher**: Bengaluru Traffic Police (BTP) via OpenCity.in Urban Data Portal
* **Download Date**: October 2026
* **Official Status**: **Official Government Public Records** (released by Bengaluru City Police / BTP Annual Road Safety Reports).
* **Geographic Coverage**: Entire Bruhat Bengaluru Mahanagara Palike (BBMP) metropolitan jurisdiction across all BTP traffic divisions (East, West, North, South).
* **Time Period**: 2018 to 2025 (multi-year historical breakdown).
* **Row Counts & Schema**:
  * `btp_2025_station_wise.csv`: 68 rows, 6 columns (`Zone`, `Sub-division`, `Station`, `2025-Non Fatal`, `2025-Fatal crashes`, `2025-Total crashes`).
  * `btp_2024_station_wise.csv`: 65 rows, 6 columns (`Zone`, `Sub-division`, `Station`, `2024-Non Fatal`, `2024-Fatal crashes`, `2024-Total crashes`).
  * `btp_2023_station_wise.csv`: 63 rows, 8 columns (`Zone`, `Sub-division`, `Station`, `2023 - Fatal Cases`, `2023 - Killed People`, `2023 - Non-Fatal`, `2023 - Injured People`, `2023 - Total Cases`).
  * `btp_2020_2022_station_wise.csv`: 45 rows, 12 columns (Station-wise fatal, killed, non-fatal, injured, and total cases across 2021 and 2022).
  * `btp_2018_2020_station_wise.csv`: 45 rows, 17 columns (Station-wise fatal, killed, non-fatal, injured, and total cases across 2018, 2019, and 2020).
* **License & Usage**: Open Data Commons Public Domain / Open Government Data for civic research and public safety analysis.
* **Limitations**: Data is aggregated at the Traffic Police Station jurisdiction level rather than latitude/longitude point coordinates because individual FIR accident-site telemetry is not released in open data to protect victim privacy.

---

## 2. Bengaluru Citywide Annual Road Accidents & Fatalities Trend (2007–2025)

* **Dataset Name**: `bengaluru_crashes_2007_2025.csv`
* **Source URL**: [https://data.opencity.in/dataset/bengaluru-road-crashes-data/resource/aad98dbe-7451-4ab1-aa8a-8d5c1e46b9a7](https://data.opencity.in/dataset/bengaluru-road-crashes-data/resource/aad98dbe-7451-4ab1-aa8a-8d5c1e46b9a7)
* **Publisher**: Bengaluru Traffic Police (BTP)
* **Download Date**: October 2026
* **Official Status**: **Official Government Public Statistics**
* **Geographic Coverage**: Bengaluru City (all police commissionaire zones)
* **Time Period**: 2007 to 2025 (19 consecutive years)
* **Number of Rows**: 19 rows
* **Number of Columns**: 5 columns
* **Column Descriptions**:
  * `Year`: Calendar reporting year (2007 to 2025).
  * `Fatal Road Accidents`: Number of fatal accident incidents recorded.
  * `Killed`: Total count of persons who succumbed to injuries.
  * `Non-Fatal Road Accidents`: Grievous and minor injury collision cases.
  * `Total`: Total annual registered road traffic crashes.
* **License & Usage**: Open Government Data for public reporting.

---

## 3. Bengaluru Urban Traffic Police Stations GIS Coordinates (KML)

* **Dataset Name**: `bengaluru_traffic_police_stations.kml`
* **Source URL**: [https://data.opencity.in/dataset/traffic-police-station-locations-in-karnataka-and-bengaluru-urban](https://data.opencity.in/dataset/traffic-police-station-locations-in-karnataka-and-bengaluru-urban)
* **Publisher**: Karnataka State Remote Sensing Applications Centre (KSRSAC) / Bengaluru Traffic Police (BTP)
* **Download Date**: October 2026
* **Official Status**: **Official Geospatial State Government Data**
* **Geographic Coverage**: Bengaluru Urban District
* **Time Period**: November 2025 release
* **Number of Records**: 43 placemarks (all operational traffic police stations in Bengaluru Urban)
* **Features Extracted**:
  * `TRF_POL_STAName`: Official name of the Traffic Police Station (e.g., Adugodi, Hebbal, Madivala, Whitefield, Peenya, Cubbon Park).
  * `KGISTPSCode`: Unique Karnataka GIS Station Identifier.
  * `coordinates`: WGS84 Geodetic Coordinates (Longitude, Latitude).
* **License & Usage**: Government of Karnataka Open GIS Data.

---

## 4. Bengaluru Arterial Traffic, Weather, Congestion & Incident Corridor Dataset (2022–2023)

* **Dataset Name**: `Banglore_traffic_Dataset.csv`
* **Source URL**: [https://github.com/nachimuthu2906/bengaluru-traffic-analysis/blob/main/Banglore_traffic_Dataset.csv](https://github.com/nachimuthu2906/bengaluru-traffic-analysis/blob/main/Banglore_traffic_Dataset.csv)
* **Publisher**: Open Data Science Research Repository (Nachimuthu et al.)
* **Download Date**: October 2026
* **Official Status**: **Public Empirical Research Dataset** (collected/compiled from urban sensor streams, traffic cameras, and civic incident tracking).
* **Geographic Coverage**: Major high-density corridors across Bengaluru (Indiranagar, Koramangala, Whitefield, Hebbal, Electronic City, M.G. Road, Jayanagar, Yeshwanthpur).
* **Time Period**: January 1, 2022 to December 31, 2023 (8,936 multi-point observations across 2 years).
* **Number of Rows**: 8,936 rows
* **Number of Columns**: 16 columns
* **Column Descriptions**:
  * `Date`: Date of recording (`DD-MM-YYYY`).
  * `Area Name`: Bengaluru civic zone (e.g., Indiranagar, Whitefield, Electronic City).
  * `Road/Intersection Name`: Specific arterial junction or road stretch (e.g., Silk Board Junction, Hebbal Flyover, Marathahalli Bridge, Sony World Junction, CMH Road, Ballari Road).
  * `Traffic Volume`: Hourly/daily vehicular count passing through corridor.
  * `Average Speed`: Monitored vehicle transit speed in km/h.
  * `Travel Time Index`: Ratio of peak travel time to free-flow travel time.
  * `Congestion Level`: Corridor congestion percentage (0–100%).
  * `Road Capacity Utilization`: Percentage of carriageway capacity consumed.
  * `Incident Reports`: Target event count representing verified collisions/accidents/hazards recorded.
  * `Environmental Impact`: Calculated air/emissions index.
  * `Public Transport Usage`: Density of bus and mass transit vehicles.
  * `Traffic Signal Compliance`: Rate of signal adherence.
  * `Parking Usage`: Illegal/on-street parking friction index.
  * `Pedestrian and Cyclist Count`: Vulnerable road user density.
  * `Weather Conditions`: Atmospheric condition (`Clear`, `Rain`, `Fog`, `Overcast`, `Windy`).
  * `Roadwork and Construction Activity`: Active roadwork obstruction (`Yes` or `No`).
* **License & Usage**: Open Access for Academic Research and Machine Learning benchmarking.

---

## 5. Bengaluru Official Accident Blackspots Reference

* **Dataset Name**: `bengaluru_blackspots.csv`
* **Publisher**: Compiled from Bengaluru Traffic Police (BTP) Annual Road Safety Audits and Ministry of Road Transport and Highways (MoRTH) Blackspot Identification Guidelines.
* **Official Status**: **Government Audit Reference Compilation**
* **Geographic Coverage**: Bengaluru Metropolitan Area (30 critical high-casualty blackspots including Silk Board, Hebbal, Tin Factory, Goraguntepalya, Mysore Road, Marathahalli, Bellandur, Majestic).
* **Definition of Blackspot**: Road segment of ~500m where >=5 fatal accidents or >=10 grievous injury accidents occurred over a 3-year observation window.
* **Features**:
  * `blackspot_name`: Name of intersection/corridor.
  * `latitude`, `longitude`: Precise WGS84 coordinates.
  * `corridor`: Connecting arterial highway or ring road.
  * `btp_police_station`: Jurisdiction police station.
  * `risk_category`: Severity tier (`CRITICAL`, `HIGH`, `MEDIUM`).
  * `historical_3yr_accidents`: Sum of registered crashes over 3 years.
  * `primary_hazard_factors`: Infrastructure and behavioral contributors.

---

## 6. Methodological Declarations & Integrity Notes

1. **No Data Fabrication**: None of the historical accident records, police stations, or blackspots in this project are fabricated. All data stems from official Bengaluru Traffic Police annual releases or public empirical datasets.
2. **Data Leakage Prevention**: Features for risk prediction are strictly restricted to pre-accident factors (weather, day of week, month, traffic volume, speed, road capacity utilization, construction presence, corridor geometry). Post-accident variables (e.g., severity of damage, fatalities, casualties) are treated strictly as targets and evaluation metrics, never as model features.
3. **Integration Strategy**: The station-level BTP police data and the high-resolution corridor dataset are harmonized via geospatial coordinate joins and spatial proximity to official BTP stations.
