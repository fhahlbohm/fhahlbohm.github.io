// Written by Florian Hahlbohm, November 2024.

// setup
const inpcFilenames = ["assets/images/m360_bicycle_00001_inpc.avif", "assets/images/m360_bicycle_00003_inpc.avif", "assets/images/m360_bicycle_00014_inpc.avif", "assets/images/m360_bonsai_00001_inpc.avif", "assets/images/m360_bonsai_00003_inpc.avif", "assets/images/m360_bonsai_00015_inpc.avif", "assets/images/m360_counter_00001_inpc.avif", "assets/images/m360_counter_00014_inpc.avif", "assets/images/m360_counter_00019_inpc.avif", "assets/images/m360_flowers_00001_inpc.avif", "assets/images/m360_flowers_00013_inpc.avif", "assets/images/m360_flowers_00020_inpc.avif", "assets/images/m360_garden_00002_inpc.avif", "assets/images/m360_garden_00009_inpc.avif", "assets/images/m360_garden_00015_inpc.avif", "assets/images/m360_kitchen_00001_inpc.avif", "assets/images/m360_kitchen_00021_inpc.avif", "assets/images/m360_kitchen_00031_inpc.avif", "assets/images/m360_room_00002_inpc.avif", "assets/images/m360_room_00007_inpc.avif", "assets/images/m360_room_00029_inpc.avif", "assets/images/m360_room_00035_inpc.avif", "assets/images/m360_stump_00000_inpc.avif", "assets/images/m360_stump_00004_inpc.avif", "assets/images/m360_stump_00008_inpc.avif", "assets/images/m360_treehill_00000_inpc.avif", "assets/images/m360_treehill_00007_inpc.avif", "assets/images/m360_treehill_00014_inpc.avif", "assets/images/tandt_family_00004_inpc.avif", "assets/images/tandt_family_00008_inpc.avif", "assets/images/tandt_family_00017_inpc.avif", "assets/images/tandt_francis_00005_inpc.avif", "assets/images/tandt_francis_00023_inpc.avif", "assets/images/tandt_francis_00037_inpc.avif", "assets/images/tandt_horse_00002_inpc.avif", "assets/images/tandt_horse_00013_inpc.avif", "assets/images/tandt_horse_00014_inpc.avif", "assets/images/tandt_lighthouse_00006_inpc.avif", "assets/images/tandt_lighthouse_00010_inpc.avif", "assets/images/tandt_lighthouse_00018_inpc.avif", "assets/images/tandt_lighthouse_00031_inpc.avif", "assets/images/tandt_m60_00002_inpc.avif", "assets/images/tandt_m60_00017_inpc.avif", "assets/images/tandt_m60_00027_inpc.avif", "assets/images/tandt_m60_00029_inpc.avif", "assets/images/tandt_m60_00037_inpc.avif", "assets/images/tandt_panther_00008_inpc.avif", "assets/images/tandt_panther_00032_inpc.avif", "assets/images/tandt_panther_00036_inpc.avif", "assets/images/tandt_playground_00003_inpc.avif", "assets/images/tandt_playground_00011_inpc.avif", "assets/images/tandt_playground_00038_inpc.avif", "assets/images/tandt_train_00005_inpc.avif", "assets/images/tandt_train_00019_inpc.avif", "assets/images/tandt_train_00021_inpc.avif"];
const zipnerfFilenames = ["assets/images/m360_bicycle_00001_zipnerf.avif", "assets/images/m360_bicycle_00003_zipnerf.avif", "assets/images/m360_bicycle_00014_zipnerf.avif", "assets/images/m360_bonsai_00001_zipnerf.avif", "assets/images/m360_bonsai_00003_zipnerf.avif", "assets/images/m360_bonsai_00015_zipnerf.avif", "assets/images/m360_counter_00001_zipnerf.avif", "assets/images/m360_counter_00014_zipnerf.avif", "assets/images/m360_counter_00019_zipnerf.avif", "assets/images/m360_flowers_00001_zipnerf.avif", "assets/images/m360_flowers_00013_zipnerf.avif", "assets/images/m360_flowers_00020_zipnerf.avif", "assets/images/m360_garden_00002_zipnerf.avif", "assets/images/m360_garden_00009_zipnerf.avif", "assets/images/m360_garden_00015_zipnerf.avif", "assets/images/m360_kitchen_00001_zipnerf.avif", "assets/images/m360_kitchen_00021_zipnerf.avif", "assets/images/m360_kitchen_00031_zipnerf.avif", "assets/images/m360_room_00002_zipnerf.avif", "assets/images/m360_room_00007_zipnerf.avif", "assets/images/m360_room_00029_zipnerf.avif", "assets/images/m360_room_00035_zipnerf.avif", "assets/images/m360_stump_00000_zipnerf.avif", "assets/images/m360_stump_00004_zipnerf.avif", "assets/images/m360_stump_00008_zipnerf.avif", "assets/images/m360_treehill_00000_zipnerf.avif", "assets/images/m360_treehill_00007_zipnerf.avif", "assets/images/m360_treehill_00014_zipnerf.avif", "assets/images/tandt_family_00004_zipnerf.avif", "assets/images/tandt_family_00008_zipnerf.avif", "assets/images/tandt_family_00017_zipnerf.avif", "assets/images/tandt_francis_00005_zipnerf.avif", "assets/images/tandt_francis_00023_zipnerf.avif", "assets/images/tandt_francis_00037_zipnerf.avif", "assets/images/tandt_horse_00002_zipnerf.avif", "assets/images/tandt_horse_00013_zipnerf.avif", "assets/images/tandt_horse_00014_zipnerf.avif", "assets/images/tandt_lighthouse_00006_zipnerf.avif", "assets/images/tandt_lighthouse_00010_zipnerf.avif", "assets/images/tandt_lighthouse_00018_zipnerf.avif", "assets/images/tandt_lighthouse_00031_zipnerf.avif", "assets/images/tandt_m60_00002_zipnerf.avif", "assets/images/tandt_m60_00017_zipnerf.avif", "assets/images/tandt_m60_00027_zipnerf.avif", "assets/images/tandt_m60_00029_zipnerf.avif", "assets/images/tandt_m60_00037_zipnerf.avif", "assets/images/tandt_panther_00008_zipnerf.avif", "assets/images/tandt_panther_00032_zipnerf.avif", "assets/images/tandt_panther_00036_zipnerf.avif", "assets/images/tandt_playground_00003_zipnerf.avif", "assets/images/tandt_playground_00011_zipnerf.avif", "assets/images/tandt_playground_00038_zipnerf.avif", "assets/images/tandt_train_00005_zipnerf.avif", "assets/images/tandt_train_00019_zipnerf.avif", "assets/images/tandt_train_00021_zipnerf.avif"];
const imagePairs = inpcFilenames.map((img, idx) => ({inpc: img, zipnerf: zipnerfFilenames[idx]}));
const numPairs = imagePairs.length;
document.getElementById("progress-bar-text").innerText = "0 / " + numPairs;
let currentPair = 0;
let wasPairSwapped = false;
let inpcCount = 0;
let zipnerfCount = 0;

// From https://stackoverflow.com/a/12646864
function shuffleArray(array) {
    for (let i = array.length - 1; i >= 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

function toggleResultsBlur() {
    document.getElementById('results').classList.toggle('visible');
}

function loadImagePair() {
    if (currentPair < imagePairs.length) {
        const {inpc, zipnerf} = imagePairs[currentPair];
        wasPairSwapped = Math.random() < 0.5;
        if (wasPairSwapped) {
            document.getElementById("left-image").src = zipnerf;
            document.getElementById("right-image").src = inpc;
        }
        else {
            document.getElementById("left-image").src = inpc;
            document.getElementById("right-image").src = zipnerf;
        }
        document.getElementById("left-image").title = "";
        document.getElementById("right-image").title = "";
    }
    else endSurvey();
}

function selectImage(side) {
    if ((side === 0 && !wasPairSwapped) || (side === 1 && wasPairSwapped)) inpcCount++;
    else zipnerfCount++;
    currentPair++;
    document.getElementById("inpc-count").innerText = inpcCount.toString();
    document.getElementById("zipnerf-count").innerText = zipnerfCount.toString();
    document.getElementById("inpc-percentage").innerText = (inpcCount / currentPair * 100).toFixed(1) + "%";
    document.getElementById("zipnerf-percentage").innerText = (zipnerfCount / currentPair * 100).toFixed(1) + "%";
    document.getElementById("progress-bar").style.width = (currentPair / numPairs * 100).toFixed(1) + "%";
    document.getElementById("progress-bar-text").innerText = currentPair + " / " + numPairs;
    loadImagePair();
}

function endSurvey() {
    document.getElementById("inpc-count").innerText = inpcCount.toString();
    document.getElementById("zipnerf-count").innerText = zipnerfCount.toString();
    document.getElementById("inpc-percentage").innerText = (inpcCount / numPairs * 100).toFixed(1) + "%";
    document.getElementById("zipnerf-percentage").innerText = (zipnerfCount / numPairs * 100).toFixed(1) + "%";
    document.getElementById("images-row").style.display = "none";
    document.getElementById("toggle-results-group").style.display = "none";
    document.getElementById('results').classList.add('visible');
}

// We want image pairs to be shown in random order.
shuffleArray(imagePairs);

// Load the first image pair.
loadImagePair();
